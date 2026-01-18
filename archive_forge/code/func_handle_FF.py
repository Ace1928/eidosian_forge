import code
import sys
import tokenize
from io import BytesIO
from traceback import format_exception
from types import TracebackType
from typing import Type
from twisted.conch import recvline
from twisted.internet import defer
from twisted.python.compat import _get_async_param
from twisted.python.htmlizer import TokenPrinter
from twisted.python.monkey import MonkeyPatcher
def handle_FF(self):
    """
        Handle a 'form feed' byte - generally used to request a screen
        refresh/redraw.
        """
    self.terminal.eraseDisplay()
    self.terminal.cursorHome()
    self.drawInputLine()