import copyreg as copy_reg
import re
import types
from twisted.persisted import crefutil
from twisted.python import log, reflect
from twisted.python.compat import _constructMethod
from ._tokenize import generate_tokens as tokenize
def setRef(self, num):
    if self.refnum:
        raise ValueError(f'Error setting id {num}, I already have {self.refnum}')
    self.refnum = num