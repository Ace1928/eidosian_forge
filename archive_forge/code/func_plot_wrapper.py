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
@contextmanager
def plot_wrapper(plot, plot_file, plot_title, train_dirs):
    if plot:
        widget = _get_catboost_widget(train_dirs)
        widget._run_update()
    try:
        yield
    finally:
        if plot:
            widget._stop_update()
    if plot_file is not None:
        OfflineMetricVisualizer(train_dirs).save_to_file(plot_title, plot_file)