import string
from typing import Dict
from zope.interface import implementer
from twisted.conch.insults import helper, insults
from twisted.logger import Logger
from twisted.python import reflect
from twisted.python.compat import iterbytes
def handle_INSERT(self):
    assert self.mode in ('typeover', 'insert')
    if self.mode == 'typeover':
        self.setInsertMode()
    else:
        self.setTypeoverMode()