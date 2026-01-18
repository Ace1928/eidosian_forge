import sys
import warnings
from gi.repository import GObject
from .._ossighelper import wakeup_on_signal, register_sigint_fallback
from .._gtktemplate import Template, _extract_handler_and_args
from ..overrides import (override, strip_boolean_result, deprecated_init,
from ..module import get_introspection_module
from gi import PyGIDeprecationWarning
def set_sort_func(self, sort_func, user_data=None):
    if sort_func is not None:
        compare_func = wrap_list_store_sort_func(sort_func)
    else:
        compare_func = None
    return super(CustomSorter, self).set_sort_func(compare_func, user_data)