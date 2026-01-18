import sys
from collections.abc import Iterable
from functools import partial
from wandb_promise import Promise, is_thenable
from ...error import GraphQLError, GraphQLLocatedError
from ...type import (GraphQLEnumType, GraphQLInterfaceType, GraphQLList,
from ..base import default_resolve_fn
from ...execution import executor
from .utils import imap, normal_map
def complete_list_value(inner_resolver, exe_context, info, on_error, result):
    if result is None:
        return None
    assert isinstance(result, Iterable), ('User Error: expected iterable, but did not find one ' + 'for field {}.{}.').format(info.parent_type, info.field_name)
    completed_results = normal_map(inner_resolver, result)
    if not any(imap(is_thenable, completed_results)):
        return completed_results
    return Promise.all(completed_results).catch(on_error)