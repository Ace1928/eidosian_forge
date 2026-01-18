from __future__ import unicode_literals
from .enums import DEFAULT_BUFFER, SEARCH_BUFFER, SYSTEM_BUFFER, DUMMY_BUFFER
from .buffer import Buffer, AcceptAction
from .history import InMemoryHistory
import six
def current_name(self, cli):
    """
        The name of the active :class:`.Buffer`.
        """
    return self.focus_stack[-1]