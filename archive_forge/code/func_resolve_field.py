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
def resolve_field(exe_context, parent_type, source, field_asts):
    field_ast = field_asts[0]
    field_name = field_ast.name.value
    field_def = get_field_def(exe_context.schema, parent_type, field_name)
    if not field_def:
        return Undefined
    return_type = field_def.type
    resolve_fn = field_def.resolver or default_resolve_fn
    resolve_fn_middleware = exe_context.get_field_resolver(resolve_fn)
    args = exe_context.get_argument_values(field_def, field_ast)
    context = exe_context.context_value
    info = ResolveInfo(field_name, field_asts, return_type, parent_type, schema=exe_context.schema, fragments=exe_context.fragments, root_value=exe_context.root_value, operation=exe_context.operation, variable_values=exe_context.variable_values)
    executor = exe_context.executor
    result = resolve_or_error(resolve_fn_middleware, source, args, context, info, executor)
    return complete_value_catching_error(exe_context, return_type, field_asts, info, result)