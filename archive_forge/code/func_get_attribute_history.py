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
def get_attribute_history(self, state, key, passive=attributes.PASSIVE_NO_INITIALIZE):
    """Facade to attributes.get_state_history(), including
        caching of results."""
    hashkey = ('history', state, key)
    if hashkey in self.attributes:
        history, state_history, cached_passive = self.attributes[hashkey]
        if not cached_passive & attributes.SQL_OK and passive & attributes.SQL_OK:
            impl = state.manager[key].impl
            history = impl.get_history(state, state.dict, attributes.PASSIVE_OFF | attributes.LOAD_AGAINST_COMMITTED | attributes.NO_RAISE)
            if history and impl.uses_objects:
                state_history = history.as_state()
            else:
                state_history = history
            self.attributes[hashkey] = (history, state_history, passive)
    else:
        impl = state.manager[key].impl
        history = impl.get_history(state, state.dict, passive | attributes.LOAD_AGAINST_COMMITTED | attributes.NO_RAISE)
        if history and impl.uses_objects:
            state_history = history.as_state()
        else:
            state_history = history
        self.attributes[hashkey] = (history, state_history, passive)
    return state_history