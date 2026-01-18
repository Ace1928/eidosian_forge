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
def get_mnist_ubyte():
    """Downloads ubyte version of the MNIST dataset into a directory in the current directory
    with the name `data` and extracts all files in the zip archive to this directory.
    """
    if not os.path.isdir('data'):
        os.makedirs('data')
    if not os.path.exists('data/train-images-idx3-ubyte') or not os.path.exists('data/train-labels-idx1-ubyte') or (not os.path.exists('data/t10k-images-idx3-ubyte')) or (not os.path.exists('data/t10k-labels-idx1-ubyte')):
        zip_file_path = download('http://data.mxnet.io/mxnet/data/mnist.zip', dirname='data')
        with zipfile.ZipFile(zip_file_path) as zf:
            zf.extractall('data')