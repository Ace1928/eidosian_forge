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
def presort_deletes(self, uowcommit, states):
    if not self.passive_deletes:
        for state in states:
            uowcommit.get_attribute_history(state, self.key, self._passive_delete_flag)