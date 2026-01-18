import threading
import warnings
import ctypes
from .base import classproperty, with_metaclass, _MXClassPropertyMetaClass
from .base import _LIB
from .base import check_call
@default_ctx.setter
def default_ctx(cls, val):
    warnings.warn('Context.default_ctx has been deprecated. Please use Context.current_context() instead. Please use test_utils.set_default_context to set a default context', DeprecationWarning)
    cls._default_ctx.value = val