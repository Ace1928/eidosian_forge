from typing import cast, Dict, Optional, Set, Tuple, Type, Union
from ._events import *
from ._util import LocalProtocolError, Sentinel
def process_client_switch_proposal(self, switch_event: Type[Sentinel]) -> None:
    self.pending_switch_proposals.add(switch_event)
    self._fire_state_triggered_transitions()