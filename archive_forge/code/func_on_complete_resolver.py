import sys
from collections.abc import Iterable
from functools import partial
from wandb_promise import Promise, is_thenable
from ...error import GraphQLError, GraphQLLocatedError
from ...type import (GraphQLEnumType, GraphQLInterfaceType, GraphQLList,
from ..base import default_resolve_fn
from ...execution import executor
from .utils import imap, normal_map
def on_complete_resolver(on_error, __func, exe_context, info, __resolver, *args, **kwargs):
    try:
        result = __resolver(*args, **kwargs)
        if isinstance(result, Exception):
            return on_error(result)
        if is_thenable(result):

            def on_resolve(value):
                if isinstance(value, Exception):
                    return on_error(value)
                return value
            return result.then(on_resolve).then(__func).catch(on_error)
        return __func(result)
    except Exception as e:
        return on_error(e)