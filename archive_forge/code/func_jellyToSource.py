import copyreg as copy_reg
import re
import types
from twisted.persisted import crefutil
from twisted.python import log, reflect
from twisted.python.compat import _constructMethod
from ._tokenize import generate_tokens as tokenize
def jellyToSource(obj, file=None):
    """
    Pass me an object and, optionally, a file object.
    I'll convert the object to an AOT either return it (if no file was
    specified) or write it to the file.
    """
    aot = jellyToAOT(obj)
    if file:
        file.write(getSource(aot).encode('utf-8'))
    else:
        return getSource(aot)