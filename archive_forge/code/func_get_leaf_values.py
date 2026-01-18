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
def get_leaf_values(self):
    """
        Returns
        -------
        leaf_values : 1d-array of leaf values for all trees.
        Value corresponding to j-th leaf of i-th tree is at position
        sum(get_tree_leaf_counts()[:i]) + j (leaf and tree indexing starts from zero).
        """
    return self._object._get_leaf_values()