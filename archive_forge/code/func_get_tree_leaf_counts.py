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
def get_tree_leaf_counts(self):
    """
        Returns
        -------
        tree_leaf_counts : 1d-array of numpy.uint32 of size tree_count_.
        tree_leaf_counts[i] equals to the number of leafs in i-th tree of the ensemble.
        """
    return self._object._get_tree_leaf_counts()