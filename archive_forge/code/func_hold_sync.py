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
@contextmanager
def hold_sync(self):
    """Hold syncing any state until the outermost context manager exits"""
    if self._holding_sync is True:
        yield
    else:
        try:
            self._holding_sync = True
            yield
        finally:
            self._holding_sync = False
            self.send_state(self._states_to_send)
            self._states_to_send.clear()