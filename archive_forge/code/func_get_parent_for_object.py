import importlib
from threading import Lock
import gi
from ._gi import \
from .types import \
from ._constants import \
def get_parent_for_object(object_info):
    parent_object_info = object_info.get_parent()
    if not parent_object_info:
        gtype = object_info.get_g_type()
        if gtype and gtype.pytype:
            return gtype.pytype
        if object_info.get_fundamental() and gtype.is_instantiatable():
            return Fundamental
        return object
    namespace = parent_object_info.get_namespace()
    name = parent_object_info.get_name()
    module = importlib.import_module('gi.repository.' + namespace)
    return getattr(module, name)