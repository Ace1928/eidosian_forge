import copy
import pickle
import sys
import typing
import warnings
from types import FunctionType
from traitlets.log import get_logger
from traitlets.utils.importstring import import_item
def can_dict(obj):
    """can the *values* of a dict"""
    if istype(obj, dict):
        newobj = {}
        for k, v in obj.items():
            newobj[k] = can(v)
        return newobj
    return obj