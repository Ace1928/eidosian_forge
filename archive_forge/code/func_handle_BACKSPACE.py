import string
from typing import Dict
from zope.interface import implementer
from twisted.conch.insults import helper, insults
from twisted.logger import Logger
from twisted.python import reflect
from twisted.python.compat import iterbytes
def handle_BACKSPACE(self):
    if self.lineBufferIndex > 0:
        self.lineBufferIndex -= 1
        del self.lineBuffer[self.lineBufferIndex]
        self.terminal.cursorBackward()
        self.terminal.deleteCharacter()