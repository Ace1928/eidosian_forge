import copyreg as copy_reg
import re
import types
from twisted.persisted import crefutil
from twisted.python import log, reflect
from twisted.python.compat import _constructMethod
from ._tokenize import generate_tokens as tokenize
def callAfter(self, callable, result):
    if isinstance(result, crefutil.NotKnown):
        listResult = [None]
        result.addDependant(listResult, 1)
    else:
        listResult = [result]
    self.afterUnjelly.append((callable, listResult))