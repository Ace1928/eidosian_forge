import os
import sys
import typing
from contextlib import contextmanager
from collections.abc import Iterable
from IPython import get_ipython
from traitlets import (
from json import loads as jsonloads, dumps as jsondumps
from .. import comm
from base64 import standard_b64encode
from .utils import deprecation, _get_frame
from .._version import __protocol_version__, __control_protocol_version__, __jupyter_widgets_base_version__
import inspect
def send_state(self, key=None):
    """Sends the widget state, or a piece of it, to the front-end, if it exists.

        Parameters
        ----------
        key : unicode, or iterable (optional)
            A single property's name or iterable of property names to sync with the front-end.
        """
    state = self.get_state(key=key)
    if len(state) > 0:
        if self._property_lock:
            for name, value in state.items():
                if name in self._property_lock:
                    self._property_lock[name] = value
        state, buffer_paths, buffers = _remove_buffers(state)
        msg = {'method': 'update', 'state': state, 'buffer_paths': buffer_paths}
        self._send(msg, buffers=buffers)