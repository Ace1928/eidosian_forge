from typing import cast, Dict, Optional, Set, Tuple, Type, Union
from ._events import *
from ._util import LocalProtocolError, Sentinel
def process_keep_alive_disabled(self) -> None:
    self.keep_alive = False
    self._fire_state_triggered_transitions()