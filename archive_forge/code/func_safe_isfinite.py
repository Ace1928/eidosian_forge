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
def safe_isfinite(val):
    if val is None:
        return False
    try:
        return math.isfinite(val)
    except (TypeError, ValueError):
        pass
    try:
        return np.isfinite(val) if np.isscalar(val) else True
    except TypeError:
        return True