from collections import namedtuple
from threading import Lock
from ._authorizer import ClosedAuthorizer
from ._identity import NoIdentities
from ._error import (
import macaroonbakery.checkers as checkers
import pyrfc3339
def final(self):
    if self._expiry is not None:
        self._conds.append(checkers.time_before_caveat(self._expiry).condition)
    return sorted(set(self._conds))