import sys
from winappdbg import win32
from winappdbg import compat
from winappdbg.util import StaticClass
import re
import time
import struct
import traceback
def log_event(self, event, text=None):
    """
        Log lines of text associated with a debug event.

        @type  event: L{Event}
        @param event: Event object.

        @type  text: str
        @param text: (Optional) Text to log. If no text is provided the default
            is to show a description of the event itself.
        """
    self.__do_log(DebugLog.log_event(event, text))