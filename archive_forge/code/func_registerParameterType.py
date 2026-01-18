import re
import warnings
import weakref
from collections import OrderedDict
from .. import functions as fn
from ..Qt import QtCore
from .ParameterItem import ParameterItem
def registerParameterType(name, cls, override=False):
    """Register a parameter type in the parametertree system.

    This enables construction of custom Parameter classes by name in
    :meth:`~pyqtgraph.parametertree.Parameter.create`.
    """
    global PARAM_TYPES
    if name in PARAM_TYPES and (not override):
        raise ValueError(f"Parameter type '{name}' already exists (use override=True to replace)")
    PARAM_TYPES[name] = cls
    PARAM_NAMES[cls] = name