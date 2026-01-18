from ast import literal_eval
import json
from ipywidgets import register, CallbackDispatcher, DOMWidget
from traitlets import Any, Bool, Int, Unicode
from ..data_utils.binary_transfer import data_buffer_serialization
from ._frontend import module_name, module_version
from .debounce import debounce
def on_view_state_change(self, callback, debounce_seconds=0.2, remove=False):
    callback = debounce(debounce_seconds)(callback) if debounce_seconds > 0 else callback
    self._view_state_handlers.register_callback(callback, remove=remove)