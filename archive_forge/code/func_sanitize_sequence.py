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
def sanitize_sequence(data):
    """
    Convert dictview objects to list. Other inputs are returned unchanged.
    """
    return list(data) if isinstance(data, collections.abc.MappingView) else data