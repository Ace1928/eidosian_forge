import copy
import copyreg as copy_reg
import inspect
import pickle
import types
from io import StringIO as _cStringIO
from typing import Dict
from twisted.python import log, reflect
from twisted.python.compat import _PYPY
def pickleStringO(stringo):
    """
    Reduce the given cStringO.

    This is only called on Python 2, because the cStringIO module only exists
    on Python 2.

    @param stringo: The string output to pickle.
    @type stringo: C{cStringIO.OutputType}
    """
    'support function for copy_reg to pickle StringIO.OutputTypes'
    return (unpickleStringO, (stringo.getvalue(), stringo.tell()))