import copyreg as copy_reg
import re
import types
from twisted.persisted import crefutil
from twisted.python import log, reflect
from twisted.python.compat import _constructMethod
from ._tokenize import generate_tokens as tokenize
def setObj(self, obj):
    if self.obj:
        raise ValueError(f'Error setting obj {obj}, I already have {self.obj}')
    self.obj = obj