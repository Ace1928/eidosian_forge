import warnings
import sys
import socket
from .._ossighelper import wakeup_on_signal, register_sigint_fallback
from ..module import get_introspection_module
from .._gi import (variant_type_from_string, source_new,
from ..overrides import override, deprecated, deprecated_attr
from gi import PyGIDeprecationWarning, version_info
from gi import _option as option
from gi import _gi
from gi._error import GError
def child_watch_add(*args, **kwargs):
    """child_watch_add(priority, pid, function, *data)"""
    priority, pid, function, data = _child_watch_add_get_args(*args, **kwargs)
    return GLib.child_watch_add(priority, pid, function, *data)