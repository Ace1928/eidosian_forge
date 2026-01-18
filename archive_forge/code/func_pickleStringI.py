import copy
import copyreg as copy_reg
import inspect
import pickle
import types
from io import StringIO as _cStringIO
from typing import Dict
from twisted.python import log, reflect
from twisted.python.compat import _PYPY
def pickleStringI(stringi):
    """
    Reduce the given cStringI.

    This is only called on Python 2, because the cStringIO module only exists
    on Python 2.

    @param stringi: The string input to pickle.
    @type stringi: C{cStringIO.InputType}

    @return: a 2-tuple of (C{unpickleStringI}, (bytes, pointer))
    @rtype: 2-tuple of (function, (bytes, int))
    """
    return (unpickleStringI, (stringi.getvalue(), stringi.tell()))