import collections
import collections.abc
import contextlib
import functools
import gzip
import itertools
import math
import operator
import os
from pathlib import Path
import shlex
import subprocess
import sys
import time
import traceback
import types
import weakref
import numpy as np
import matplotlib
from matplotlib import _api, _c_internal_utils
def open_file_cm(path_or_file, mode='r', encoding=None):
    """Pass through file objects and context-manage path-likes."""
    fh, opened = to_filehandle(path_or_file, mode, True, encoding)
    return fh if opened else contextlib.nullcontext(fh)