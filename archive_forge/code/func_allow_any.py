from collections import namedtuple
from threading import Lock
from ._authorizer import ClosedAuthorizer
from ._identity import NoIdentities
from ._error import (
import macaroonbakery.checkers as checkers
import pyrfc3339
def allow_any(self, ctx, ops):
    """ like allow except that it will authorize as many of the
        operations as possible without requiring any to be authorized. If all
        the operations succeeded, the array will be nil.

        If any the operations failed, the returned error will be the same
        that allow would return and each element in the returned slice will
        hold whether its respective operation was allowed.

        If all the operations succeeded, the returned slice will be None.

        The returned AuthInfo will always be non-None.

        The LOGIN_OP operation is treated specially - it is always required if
        present in ops.
        @param ctx AuthContext
        @param ops an array of Op
        :return: an AuthInfo object and the auth used as an array of int.
        """
    authed, used = self._allow_any(ctx, ops)
    return (self._new_auth_info(used), authed)