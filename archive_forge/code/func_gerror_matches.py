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
def gerror_matches(self, domain, code):
    if isinstance(self.domain, str):
        self_domain_quark = GLib.quark_from_string(self.domain)
    else:
        self_domain_quark = self.domain
    return (self_domain_quark, self.code) == (domain, code)