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
def complete_value(exe_context, return_type, field_asts, info, result):
    """
    Implements the instructions for completeValue as defined in the
    "Field entries" section of the spec.

    If the field type is Non-Null, then this recursively completes the value for the inner type. It throws a field
    error if that completion returns null, as per the "Nullability" section of the spec.

    If the field type is a List, then this recursively completes the value for the inner type on each item in the
    list.

    If the field type is a Scalar or Enum, ensures the completed value is a legal value of the type by calling the
    `serialize` method of GraphQL type definition.

    If the field is an abstract type, determine the runtime type of the value and then complete based on that type.

    Otherwise, the field type expects a sub-selection set, and will complete the value by evaluating all
    sub-selections.
    """
    if is_thenable(result):
        return Promise.resolve(result).then(lambda resolved: complete_value(exe_context, return_type, field_asts, info, resolved), lambda error: Promise.rejected(GraphQLLocatedError(field_asts, original_error=error)))
    if isinstance(result, Exception):
        raise GraphQLLocatedError(field_asts, original_error=result)
    if isinstance(return_type, GraphQLNonNull):
        return complete_nonnull_value(exe_context, return_type, field_asts, info, result)
    if result is None:
        return None
    if isinstance(return_type, GraphQLList):
        return complete_list_value(exe_context, return_type, field_asts, info, result)
    if isinstance(return_type, (GraphQLScalarType, GraphQLEnumType)):
        return complete_leaf_value(return_type, result)
    if isinstance(return_type, (GraphQLInterfaceType, GraphQLUnionType)):
        return complete_abstract_value(exe_context, return_type, field_asts, info, result)
    if isinstance(return_type, GraphQLObjectType):
        return complete_object_value(exe_context, return_type, field_asts, info, result)
    assert False, u'Cannot complete value of unexpected type "{}".'.format(return_type)