import functools
import warnings
from collections import namedtuple
import gi.module
from gi.overrides import override, deprecated_attr
from gi.repository import GLib
from gi import PyGIDeprecationWarning
from gi import _propertyhelper as propertyhelper
from gi import _signalhelper as signalhelper
from gi import _gi
from gi import _option as option
def get_boxed(self):
    if not self.__g_type.is_a(TYPE_BOXED):
        warnings.warn('Calling get_boxed() on a non-boxed type deprecated', PyGIDeprecationWarning, stacklevel=2)
    return _gi._gvalue_get(self)