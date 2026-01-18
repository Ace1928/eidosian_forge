import time
import gzip
import struct
import traceback
import numbers
import sys
import os
import platform
import errno
import logging
import bz2
import zipfile
import json
from contextlib import contextmanager
from collections import OrderedDict
import numpy as np
import numpy.testing as npt
import numpy.random as rnd
import mxnet as mx
from .context import Context, current_context
from .ndarray.ndarray import _STORAGE_TYPE_STR_TO_ID
from .ndarray import array
from .symbol import Symbol
from .symbol.numpy import _Symbol as np_symbol
from .util import use_np, getenv, setenv  # pylint: disable=unused-import
from .runtime import Features
from .numpy_extension import get_cuda_compute_capability
def same_symbol_structure(sym1, sym2):
    """Compare two symbols to check if they have the same computation graph structure.
    Returns true if operator corresponding to a particular node id is same in both
    symbols for all nodes
    """
    conf = json.loads(sym1.tojson())
    nodes = conf['nodes']
    conf2 = json.loads(sym2.tojson())
    nodes2 = conf2['nodes']
    for node1, node2 in zip(nodes, nodes2):
        if node1['op'] != node2['op']:
            return False
    return True