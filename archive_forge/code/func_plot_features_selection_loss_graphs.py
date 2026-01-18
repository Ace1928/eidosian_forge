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
def plot_features_selection_loss_graphs(summary):
    result = {}
    result['features'] = plot_features_selection_loss_graph('Loss by eliminated features', 'features', 'features', summary['eliminated_features'], summary['eliminated_features_names'], summary['loss_graph'])
    if 'eliminated_features_tags' in summary:
        result['features_tags'] = plot_features_selection_loss_graph('Loss by eliminated features tags', 'features tags', 'features_tags', [], summary['eliminated_features_tags'], summary['features_tags_loss_graph'], cost_graph=summary['features_tags_cost_graph'])
    return result