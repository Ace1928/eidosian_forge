import copy
import copyreg as copy_reg
import inspect
import pickle
import types
from io import StringIO as _cStringIO
from typing import Dict
from twisted.python import log, reflect
from twisted.python.compat import _PYPY
def requireUpgrade(obj):
    """Require that a Versioned instance be upgraded completely first."""
    objID = id(obj)
    if objID in versionedsToUpgrade and objID not in upgraded:
        upgraded[objID] = 1
        obj.versionUpgrade()
        return obj