from __future__ import unicode_literals
import inspect
import os
import signal
import sys
import threading
import weakref
from wcwidth import wcwidth
from six.moves import range
def is_conemu_ansi():
    """
    True when the ConEmu Windows console is used.
    """
    return is_windows() and os.environ.get('ConEmuANSI', 'OFF') == 'ON'