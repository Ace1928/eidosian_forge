from __future__ import unicode_literals
from prompt_toolkit.filters import to_simple_filter, Condition
from prompt_toolkit.layout.screen import Size
from prompt_toolkit.renderer import Output
from prompt_toolkit.styles import ANSI_COLOR_NAMES
from six.moves import range
import array
import errno
import os
import six
def ask_for_cpr(self):
    """
        Asks for a cursor position report (CPR).
        """
    self.write_raw('\x1b[6n')
    self.flush()