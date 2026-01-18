from __future__ import annotations
from typing import Any
from typing import Dict
from typing import Optional
from typing import Set
from typing import TYPE_CHECKING
from . import attributes
from . import exc as orm_exc
from . import util as orm_util
from .. import event
from .. import util
from ..util import topological
def register_object(self, state: InstanceState[Any], isdelete: bool=False, listonly: bool=False, cancel_delete: bool=False, operation: Optional[str]=None, prop: Optional[MapperProperty]=None) -> bool:
    if not self.session._contains_state(state):
        if not state.deleted and operation is not None:
            util.warn("Object of type %s not in session, %s operation along '%s' will not proceed" % (orm_util.state_class_str(state), operation, prop))
        return False
    if state not in self.states:
        mapper = state.manager.mapper
        if mapper not in self.mappers:
            self._per_mapper_flush_actions(mapper)
        self.mappers[mapper].add(state)
        self.states[state] = (isdelete, listonly)
    elif not listonly and (isdelete or cancel_delete):
        self.states[state] = (isdelete, False)
    return True