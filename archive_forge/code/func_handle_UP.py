import string
from typing import Dict
from zope.interface import implementer
from twisted.conch.insults import helper, insults
from twisted.logger import Logger
from twisted.python import reflect
from twisted.python.compat import iterbytes
def handle_UP(self):
    if self.lineBuffer and self.historyPosition == len(self.historyLines):
        self.historyLines.append(b''.join(self.lineBuffer))
    if self.historyPosition > 0:
        self.handle_HOME()
        self.terminal.eraseToLineEnd()
        self.historyPosition -= 1
        self.lineBuffer = []
        self._deliverBuffer(self.historyLines[self.historyPosition])