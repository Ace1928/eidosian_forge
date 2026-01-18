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
def getFeatureIndices(features):
    if isinstance(features, list) or isinstance(features, tuple):
        features_idxs = [getFeatureIdx(feature) for feature in features]
    elif isinstance(features, int) or isinstance(features, str):
        features_idxs = [getFeatureIdx(features)]
    else:
        raise CatBoostError("Unsupported type for argument 'features'. Must be one of: int, string, list<string>, list<int>, tuple<int>, tuple<string>")
    return features_idxs