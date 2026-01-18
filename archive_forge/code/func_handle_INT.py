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
def handle_INT(self):
    """
        Handle ^C as an interrupt keystroke by resetting the current input
        variables to their initial state.
        """
    self.pn = 0
    self.lineBuffer = []
    self.lineBufferIndex = 0
    self.interpreter.resetBuffer()
    self.terminal.nextLine()
    self.terminal.write(b'KeyboardInterrupt')
    self.terminal.nextLine()
    self.terminal.write(self.ps[self.pn])