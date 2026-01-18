from __future__ import annotations
from . import attributes
from . import exc
from . import sync
from . import unitofwork
from . import util as mapperutil
from .interfaces import MANYTOMANY
from .interfaces import MANYTOONE
from .interfaces import ONETOMANY
from .. import exc as sa_exc
from .. import sql
from .. import util
def prop_has_changes(self, uow, states, isdelete):
    if not isdelete and self.passive_updates:
        d = self._key_switchers(uow, states)
        return bool(d)
    return False