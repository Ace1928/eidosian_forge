from __future__ import annotations
import datetime
import os
import threading
from collections import OrderedDict
from copy import deepcopy
from typing import TYPE_CHECKING, Any, Iterator
@property
def state_components(self) -> Iterator[tuple[State, Any, bool]]:
    from gradio.components import State
    for id in self._data:
        block = self.blocks.blocks[id]
        if isinstance(block, State) and id in self._state_ttl:
            time_to_live, created_at = self._state_ttl[id]
            if self.is_closed:
                time_to_live = self.STATE_TTL_WHEN_CLOSED
            value = self._data[id]
            yield (block, value, (datetime.datetime.now() - created_at).seconds > time_to_live)