import sys
from collections.abc import Iterable
from functools import partial
from wandb_promise import Promise, is_thenable
from ...error import GraphQLError, GraphQLLocatedError
from ...type import (GraphQLEnumType, GraphQLInterfaceType, GraphQLList,
from ..base import default_resolve_fn
from ...execution import executor
from .utils import imap, normal_map
def field_resolver(field, fragment=None, exe_context=None, info=None):
    resolver = field.resolver or default_resolve_fn
    if exe_context:
        resolver = exe_context.get_field_resolver(resolver)
    return type_resolver(field.type, resolver, fragment, exe_context, info, catch_error=True)