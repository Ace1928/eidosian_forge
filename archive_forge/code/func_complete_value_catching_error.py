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
def complete_value_catching_error(exe_context, return_type, field_asts, info, result):
    if isinstance(return_type, GraphQLNonNull):
        return complete_value(exe_context, return_type, field_asts, info, result)
    try:
        completed = complete_value(exe_context, return_type, field_asts, info, result)
        if is_thenable(completed):

            def handle_error(error):
                exe_context.errors.append(error)
                return None
            return completed.catch(handle_error)
        return completed
    except Exception as e:
        exe_context.errors.append(e)
        return None