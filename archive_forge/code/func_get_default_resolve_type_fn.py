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
def get_default_resolve_type_fn(value, context, info, abstract_type):
    possible_types = info.schema.get_possible_types(abstract_type)
    for type in possible_types:
        if callable(type.is_type_of) and type.is_type_of(value, context, info):
            return type