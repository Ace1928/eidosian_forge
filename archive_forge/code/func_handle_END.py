import string
from typing import Dict
from zope.interface import implementer
from twisted.conch.insults import helper, insults
from twisted.logger import Logger
from twisted.python import reflect
from twisted.python.compat import iterbytes
def handle_END(self):
    offset = len(self.lineBuffer) - self.lineBufferIndex
    if offset:
        self.terminal.cursorForward(offset)
        self.lineBufferIndex = len(self.lineBuffer)