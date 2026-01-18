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
def process_deletes(self, uowcommit, states):
    secondary_delete = []
    secondary_insert = []
    secondary_update = []
    processed = self._get_reversed_processed_set(uowcommit)
    tmp = set()
    for state in states:
        history = uowcommit.get_attribute_history(state, self.key, self._passive_delete_flag)
        if history:
            for child in history.non_added():
                if child is None or (processed is not None and (state, child) in processed):
                    continue
                associationrow = {}
                if not self._synchronize(state, child, associationrow, False, uowcommit, 'delete'):
                    continue
                secondary_delete.append(associationrow)
            tmp.update(((c, state) for c in history.non_added()))
    if processed is not None:
        processed.update(tmp)
    self._run_crud(uowcommit, secondary_insert, secondary_update, secondary_delete)