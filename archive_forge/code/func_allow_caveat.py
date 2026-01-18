import collections
import pyrfc3339
from ._conditions import (
def allow_caveat(ops):
    """ Returns a caveat that will deny attempts to use the macaroon to perform
    any operation other than those listed. Operations must not contain a space.
    """
    if ops is None or len(ops) == 0:
        return error_caveat('no operations allowed')
    return _operation_caveat(COND_ALLOW, ops)