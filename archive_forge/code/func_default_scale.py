import atexit
import json
import os
import socket
import subprocess
import sys
import threading
import warnings
from copy import copy
from contextlib import contextmanager
from pathlib import Path
from shutil import which
import tenacity
import plotly
from plotly.files import PLOTLY_DIR, ensure_writable_plotly_dir
from plotly.io._utils import validate_coerce_fig_to_dict
from plotly.optional_imports import get_module
@default_scale.setter
def default_scale(self, val):
    if val is None:
        self._props.pop('default_scale', None)
        return
    if not isinstance(val, (int, float)):
        raise ValueError('\nThe default_scale property must be a number, but received value of type {typ}.\n    Received value: {val}'.format(typ=type(val), val=val))
    self._props['default_scale'] = val