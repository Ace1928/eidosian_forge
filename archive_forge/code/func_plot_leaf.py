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
def plot_leaf(node_idx, graph):
    cur_id = 'leaf_{}'.format(node_to_leaf[node_idx])
    node_label = leaf_values[node_to_leaf[node_idx]]
    graph.node(cur_id, node_label, color='red', shape='rect')
    return cur_id