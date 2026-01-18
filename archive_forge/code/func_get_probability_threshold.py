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
def get_probability_threshold(self):
    """
        Get a threshold for class separation in binary classification task
        """
    if not self.is_fitted():
        raise CatBoostError("Not fitted models don't have a probability threshold.")
    return self._object._get_binclass_probability_threshold()