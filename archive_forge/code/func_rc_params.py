import atexit
from collections import namedtuple
from collections.abc import MutableMapping
import contextlib
import functools
import importlib
import inspect
from inspect import Parameter
import locale
import logging
import os
from pathlib import Path
import pprint
import re
import shutil
import subprocess
import sys
import tempfile
import warnings
import numpy
from packaging.version import parse as parse_version
from . import _api, _version, cbook, _docstring, rcsetup
from matplotlib.cbook import sanitize_sequence
from matplotlib._api import MatplotlibDeprecationWarning
from matplotlib.rcsetup import validate_backend, cycler
from matplotlib.cm import _colormaps as colormaps
from matplotlib.colors import _color_sequences as color_sequences
def rc_params(fail_on_error=False):
    """Construct a `RcParams` instance from the default Matplotlib rc file."""
    return rc_params_from_file(matplotlib_fname(), fail_on_error)