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
@mathjax.setter
def mathjax(self, val):
    if val is None:
        self._props.pop('mathjax', None)
    else:
        if not isinstance(val, str):
            raise ValueError('\nThe mathjax property must be a string, but received value of type {typ}.\n    Received value: {val}'.format(typ=type(val), val=val))
        self._props['mathjax'] = val
    shutdown_server()