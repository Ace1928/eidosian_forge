import string
from typing import Dict
from zope.interface import implementer
from twisted.conch.insults import helper, insults
from twisted.logger import Logger
from twisted.python import reflect
from twisted.python.compat import iterbytes
def currentHistoryBuffer(self):
    b = tuple(self.historyLines)
    return (b[:self.historyPosition], b[self.historyPosition:])