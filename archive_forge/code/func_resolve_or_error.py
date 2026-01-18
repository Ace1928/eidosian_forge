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
def resolve_or_error(resolve_fn, source, args, context, info, executor):
    try:
        return executor.execute(resolve_fn, source, args, context, info)
    except Exception as e:
        logger.exception('An error occurred while resolving field {}.{}'.format(info.parent_type.name, info.field_name))
        e.stack = sys.exc_info()[2]
        return e