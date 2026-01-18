import pyrfc3339
from ._auth_context import ContextKey
from ._caveat import parse_caveat
from ._conditions import COND_TIME_BEFORE, STD_NAMESPACE
from ._utils import condition_with_prefix
def context_with_clock(ctx, clock):
    """ Returns a copy of ctx with a key added that associates it with the
    given clock implementation, which will be used by the time-before checker
    to determine the current time.
    The clock should have a utcnow method that returns the current time
    as a datetime value in UTC.
    """
    if clock is None:
        return ctx
    return ctx.with_value(TIME_KEY, clock)