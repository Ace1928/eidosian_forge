from __future__ import unicode_literals
from .base import Filter
from prompt_toolkit.enums import EditingMode
from prompt_toolkit.key_binding.vi_state import InputMode as ViInputMode
from prompt_toolkit.cache import memoized
@property
def buffer_name(self):
    """ The given buffer name. (Read-only) """
    return self._buffer_name