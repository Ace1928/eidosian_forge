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
def getFeatureIdx(feature):
    if not isinstance(feature, int):
        if self.feature_names_ is None or feature not in self.feature_names_:
            raise CatBoostError('No feature named "{}" in model'.format(feature))
        feature_idx = self.feature_names_.index(feature)
    else:
        feature_idx = feature
    assert feature_idx in self._get_borders(), 'only float features indexes are supported'
    assert len(self._get_borders()[feature_idx]) > 0, 'feature with idx {} is not used in model'.format(feature_idx)
    return feature_idx