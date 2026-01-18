from contextlib import contextmanager  # noqa E402
from copy import deepcopy
import logging
import sys
import os
from collections import OrderedDict, defaultdict
from six import iteritems, string_types, integer_types
import warnings
import numpy as np
import ctypes
import platform
import tempfile
import shutil
import json
from enum import Enum
from operator import itemgetter
import threading
import scipy.sparse
from .plot_helpers import save_plot_file, try_plot_offline, OfflineMetricVisualizer
from . import _catboost
from .metrics import BuiltinMetric
def enum_from_enum_or_str(enum_type, arg):
    if isinstance(arg, enum_type):
        return arg
    elif isinstance(arg, str):
        return enum_type[arg]
    else:
        raise Exception("can't create enum " + str(enum_type) + ' from type ' + str(type(arg)))