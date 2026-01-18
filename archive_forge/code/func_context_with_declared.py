from ._auth_context import ContextKey
from ._caveat import Caveat, error_caveat, parse_caveat
from ._conditions import (
from ._namespace import Namespace
def context_with_declared(ctx, declared):
    """ Returns a context with attached declared information,
    as returned from infer_declared.
    """
    return ctx.with_value(DECLARED_KEY, declared)