import copy
import copyreg as copy_reg
import inspect
import pickle
import types
from io import StringIO as _cStringIO
from typing import Dict
from twisted.python import log, reflect
from twisted.python.compat import _PYPY
def pickleMethod(method):
    """support function for copy_reg to pickle method refs"""
    return (unpickleMethod, (method.__name__, method.__self__, method.__self__.__class__))