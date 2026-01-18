import inspect
from functools import partial
from ..utils.module_loading import import_string
from .mountedtype import MountedType
from .unmountedtype import UnmountedType
def get_underlying_type(_type):
    """Get the underlying type even if it is wrapped in structures like NonNull"""
    while hasattr(_type, 'of_type'):
        _type = _type.of_type
    return _type