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
def get_im2rec_path(home_env='MXNET_HOME'):
    """Get path to the im2rec.py tool

    Parameters
    ----------

    home_env : str
        Env variable that holds the path to the MXNET folder

    Returns
    -------
    str
        The path to im2rec.py
    """
    if home_env in os.environ:
        mxnet_path = os.environ[home_env]
    else:
        mxnet_path = os.path.dirname(mx.__file__)
    im2rec_path = os.path.join(mxnet_path, 'tools', 'im2rec.py')
    if os.path.isfile(im2rec_path):
        return im2rec_path
    im2rec_path = os.path.join(mxnet_path, '..', '..', 'tools', 'im2rec.py')
    if os.path.isfile(im2rec_path):
        return im2rec_path
    raise IOError('Could not find path to tools/im2rec.py')