import sys
from collections.abc import Iterable
from functools import partial
from wandb_promise import Promise, is_thenable
from ...error import GraphQLError, GraphQLLocatedError
from ...type import (GraphQLEnumType, GraphQLInterfaceType, GraphQLList,
from ..base import default_resolve_fn
from ...execution import executor
from .utils import imap, normal_map
def complete_object_value(fragment_resolve, exe_context, on_error, result):
    if result is None:
        return None
    result = fragment_resolve(result)
    if is_thenable(result):
        return result.catch(on_error)
    return result