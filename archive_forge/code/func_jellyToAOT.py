import copyreg as copy_reg
import re
import types
from twisted.persisted import crefutil
from twisted.python import log, reflect
from twisted.python.compat import _constructMethod
from ._tokenize import generate_tokens as tokenize
def jellyToAOT(obj):
    """Convert an object to an Abstract Object Tree."""
    return AOTJellier().jelly(obj)