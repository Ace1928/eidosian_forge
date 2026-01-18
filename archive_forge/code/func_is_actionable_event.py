import collections
import prettytable
from automaton import _utils as utils
from automaton import exceptions as excp
def is_actionable_event(self, event):
    """Check whether the event is actionable in the current state."""
    current = self._current
    if current is None:
        return False
    if event not in self._transitions[current.name]:
        return False
    return True