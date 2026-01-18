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
def process_saves(self, uowcommit, states):
    secondary_delete = []
    secondary_insert = []
    secondary_update = []
    processed = self._get_reversed_processed_set(uowcommit)
    tmp = set()
    for state in states:
        need_cascade_pks = not self.passive_updates and self._pks_changed(uowcommit, state)
        if need_cascade_pks:
            passive = attributes.PASSIVE_OFF | attributes.INCLUDE_PENDING_MUTATIONS
        else:
            passive = attributes.PASSIVE_NO_INITIALIZE | attributes.INCLUDE_PENDING_MUTATIONS
        history = uowcommit.get_attribute_history(state, self.key, passive)
        if history:
            for child in history.added:
                if processed is not None and (state, child) in processed:
                    continue
                associationrow = {}
                if not self._synchronize(state, child, associationrow, False, uowcommit, 'add'):
                    continue
                secondary_insert.append(associationrow)
            for child in history.deleted:
                if processed is not None and (state, child) in processed:
                    continue
                associationrow = {}
                if not self._synchronize(state, child, associationrow, False, uowcommit, 'delete'):
                    continue
                secondary_delete.append(associationrow)
            tmp.update(((c, state) for c in history.added + history.deleted))
            if need_cascade_pks:
                for child in history.unchanged:
                    associationrow = {}
                    sync.update(state, self.parent, associationrow, 'old_', self.prop.synchronize_pairs)
                    sync.update(child, self.mapper, associationrow, 'old_', self.prop.secondary_synchronize_pairs)
                    secondary_update.append(associationrow)
    if processed is not None:
        processed.update(tmp)
    self._run_crud(uowcommit, secondary_insert, secondary_update, secondary_delete)