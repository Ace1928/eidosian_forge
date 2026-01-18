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
@default_format.setter
def default_format(self, val):
    if val is None:
        self._props.pop('default_format', None)
        return
    val = validate_coerce_format(val)
    self._props['default_format'] = val