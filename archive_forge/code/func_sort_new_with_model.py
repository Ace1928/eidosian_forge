import sys
import warnings
from gi.repository import GObject
from .._ossighelper import wakeup_on_signal, register_sigint_fallback
from .._gtktemplate import Template, _extract_handler_and_args
from ..overrides import (override, strip_boolean_result, deprecated_init,
from ..module import get_introspection_module
from gi import PyGIDeprecationWarning
def sort_new_with_model(self):
    super_object = super(TreeModel, self)
    if hasattr(super_object, 'sort_new_with_model'):
        return super_object.sort_new_with_model()
    else:
        return TreeModelSort.new_with_model(self)