import sys
import warnings
from gi.repository import GObject
from .._ossighelper import wakeup_on_signal, register_sigint_fallback
from .._gtktemplate import Template, _extract_handler_and_args
from ..overrides import (override, strip_boolean_result, deprecated_init,
from ..module import get_introspection_module
from gi import PyGIDeprecationWarning
def drag_dest_set_target_list(self, target_list):
    if target_list is not None and (not isinstance(target_list, Gtk.TargetList)):
        target_list = Gtk.TargetList.new(_construct_target_list(target_list))
    super(Widget, self).drag_dest_set_target_list(target_list)