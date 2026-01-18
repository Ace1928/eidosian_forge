import re
import warnings
import weakref
from collections import OrderedDict
from .. import functions as fn
from ..Qt import QtCore
from .ParameterItem import ParameterItem
def registerParameterItemType(name, itemCls, parameterCls=None, override=False):
    """
    Similar to :func:`registerParameterType`, but works on ParameterItems. This is useful for Parameters where the
    `itemClass` does all the heavy lifting, and a redundant Parameter class must be defined just to house `itemClass`.
    Instead, use `registerParameterItemType`. If this should belong to a subclass of `Parameter`, specify which one
    in `parameterCls`.
    """
    global _PARAM_ITEM_TYPES
    if name in _PARAM_ITEM_TYPES and (not override):
        raise ValueError(f"Parameter item type '{name}' already exists (use override=True to replace)")
    parameterCls = parameterCls or Parameter
    _PARAM_ITEM_TYPES[name] = itemCls
    registerParameterType(name, parameterCls, override)