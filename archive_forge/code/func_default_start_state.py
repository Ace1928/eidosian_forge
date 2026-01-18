import collections
import prettytable
from automaton import _utils as utils
from automaton import exceptions as excp
@default_start_state.setter
def default_start_state(self, state):
    if self.frozen:
        raise excp.FrozenMachine()
    if state not in self._states:
        raise excp.NotFound("Can not set the default start state to undefined state '%s'" % state)
    self._default_start_state = state