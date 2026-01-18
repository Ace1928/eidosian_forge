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
def register_post_update(self, state, post_update_cols):
    mapper = state.manager.mapper.base_mapper
    states, cols = self.post_update_states[mapper]
    states.add(state)
    cols.update(post_update_cols)