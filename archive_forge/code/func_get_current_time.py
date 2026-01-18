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
def get_current_time(self):
    return GLib.get_real_time() * 1e-06