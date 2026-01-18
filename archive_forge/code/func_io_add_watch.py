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
def io_add_watch(*args, **kwargs):
    """io_add_watch(channel, priority, condition, func, *user_data) -> event_source_id"""
    channel, priority, condition, func, user_data = _io_add_watch_get_args(*args, **kwargs)
    return GLib.io_add_watch(channel, priority, condition, func, *user_data)