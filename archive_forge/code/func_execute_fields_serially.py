import collections
from collections.abc import Iterable
import functools
import logging
import sys
from wandb_promise import Promise, promise_for_dict, is_thenable
from ..error import GraphQLError, GraphQLLocatedError
from ..pyutils.default_ordered_dict import DefaultOrderedDict
from ..pyutils.ordereddict import OrderedDict
from ..type import (GraphQLEnumType, GraphQLInterfaceType, GraphQLList,
from .base import (ExecutionContext, ExecutionResult, ResolveInfo, Undefined,
from .executors.sync import SyncExecutor
from .experimental.executor import execute as experimental_execute
from .middleware import MiddlewareManager
def execute_fields_serially(exe_context, parent_type, source_value, fields):

    def execute_field_callback(results, response_name):
        field_asts = fields[response_name]
        result = resolve_field(exe_context, parent_type, source_value, field_asts)
        if result is Undefined:
            return results
        if is_thenable(result):

            def collect_result(resolved_result):
                results[response_name] = resolved_result
                return results
            return result.then(collect_result, None)
        results[response_name] = result
        return results

    def execute_field(prev_promise, response_name):
        return prev_promise.then(lambda results: execute_field_callback(results, response_name))
    return functools.reduce(execute_field, fields.keys(), Promise.resolve(collections.OrderedDict()))