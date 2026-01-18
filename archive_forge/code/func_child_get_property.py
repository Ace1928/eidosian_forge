import sys
import warnings
from gi.repository import GObject
from .._ossighelper import wakeup_on_signal, register_sigint_fallback
from .._gtktemplate import Template, _extract_handler_and_args
from ..overrides import (override, strip_boolean_result, deprecated_init,
from ..module import get_introspection_module
from gi import PyGIDeprecationWarning
def child_get_property(self, child, property_name, value=None):
    if value is None:
        prop = self.find_child_property(property_name)
        if prop is None:
            raise ValueError('Class "%s" does not contain child property "%s"' % (self, property_name))
        value = GObject.Value(prop.value_type)
    Gtk.Container.child_get_property(self, child, property_name, value)
    return value.get_value()