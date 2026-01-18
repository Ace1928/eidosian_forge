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
def presort_saves(self, uowcommit, states):
    if not self.passive_updates:
        for state in states:
            if self._pks_changed(uowcommit, state):
                history = uowcommit.get_attribute_history(state, self.key, attributes.PASSIVE_OFF)
    if not self.cascade.delete_orphan:
        return
    for state in states:
        history = uowcommit.get_attribute_history(state, self.key, attributes.PASSIVE_NO_INITIALIZE)
        if history:
            for child in history.deleted:
                if self.hasparent(child) is False:
                    uowcommit.register_object(child, isdelete=True, operation='delete', prop=self.prop)
                    for c, m, st_, dct_ in self.mapper.cascade_iterator('delete', child):
                        uowcommit.register_object(st_, isdelete=True)