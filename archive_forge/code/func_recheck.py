import enchant
import gettext
import logging
import re
import sys
from collections import UserList
from ._pylocales import code_to_name as _code_to_name
from ._pylocales import LanguageNotFound, CountryNotFound
from gi.repository import Gio, GLib, GObject
def recheck(self):
    """
        Rechecks the spelling of the whole text.
        """
    start, end = self._buffer.get_bounds()
    if self._batched_rechecking and end.get_offset() > _BATCHING_THRESHOLD_CHARS:
        start_mark = self._buffer.create_mark(None, start)
        self._continue_batched_recheck(start_mark)
    else:
        self.check_range(start, end, True)