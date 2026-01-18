import sys
import warnings
from gi.repository import GObject
from .._ossighelper import wakeup_on_signal, register_sigint_fallback
from .._gtktemplate import Template, _extract_handler_and_args
from ..overrides import (override, strip_boolean_result, deprecated_init,
from ..module import get_introspection_module
from gi import PyGIDeprecationWarning
def enable_model_drag_source(self, start_button_mask, targets, actions):
    target_entries = _construct_target_list(targets)
    super(TreeView, self).enable_model_drag_source(start_button_mask, target_entries, actions)