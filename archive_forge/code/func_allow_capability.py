from collections import namedtuple
from threading import Lock
from ._authorizer import ClosedAuthorizer
from ._identity import NoIdentities
from ._error import (
import macaroonbakery.checkers as checkers
import pyrfc3339
def allow_capability(self, ctx, ops):
    """Checks that the user is allowed to perform all the
        given operations. If not, a discharge error will be raised.
        If allow_capability succeeds, it returns a list of first party caveat
        conditions that must be applied to any macaroon granting capability
        to execute the operations. Those caveat conditions will not
        include any declarations contained in login macaroons - the
        caller must be careful not to mint a macaroon associated
        with the LOGIN_OP operation unless they add the expected
        declaration caveat too - in general, clients should not create
        capabilities that grant LOGIN_OP rights.

        The operations must include at least one non-LOGIN_OP operation.
        """
    nops = 0
    for op in ops:
        if op != LOGIN_OP:
            nops += 1
    if nops == 0:
        raise ValueError('no non-login operations required in capability')
    _, used = self._allow_any(ctx, ops)
    squasher = _CaveatSquasher()
    for i, is_used in enumerate(used):
        if not is_used:
            continue
        for cond in self._conditions[i]:
            squasher.add(cond)
    return squasher.final()