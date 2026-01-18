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
def get_test_eval(self):
    test_evals = self._object._get_test_evals()
    if len(test_evals) == 0:
        if self.is_fitted():
            raise CatBoostError('The model has been trained without an eval set.')
        else:
            raise CatBoostError('You should train the model first.')
    if len(test_evals) > 1:
        raise CatBoostError("With multiple eval sets use 'get_test_evals()'")
    test_eval = test_evals[0]
    return test_eval[0] if len(test_eval) == 1 else test_eval