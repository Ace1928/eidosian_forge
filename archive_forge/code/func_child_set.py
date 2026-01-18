import sys
import warnings
from gi.repository import GObject
from .._ossighelper import wakeup_on_signal, register_sigint_fallback
from .._gtktemplate import Template, _extract_handler_and_args
from ..overrides import (override, strip_boolean_result, deprecated_init,
from ..module import get_introspection_module
from gi import PyGIDeprecationWarning
def child_set(self, child, **kwargs):
    """Set a child properties on the given child to key/value pairs."""
    for name, value in kwargs.items():
        name = name.replace('_', '-')
        self.child_set_property(child, name, value)