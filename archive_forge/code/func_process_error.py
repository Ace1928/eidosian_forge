from typing import cast, Dict, Optional, Set, Tuple, Type, Union
from ._events import *
from ._util import LocalProtocolError, Sentinel
def process_error(self, role: Type[Sentinel]) -> None:
    self.states[role] = ERROR
    self._fire_state_triggered_transitions()