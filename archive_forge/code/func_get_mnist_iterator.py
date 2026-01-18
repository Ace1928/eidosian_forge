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
def get_mnist_iterator(batch_size, input_shape, num_parts=1, part_index=0):
    """Returns training and validation iterators for MNIST dataset
    """
    get_mnist_ubyte()
    flat = len(input_shape) != 3
    train_dataiter = mx.io.MNISTIter(image='data/train-images-idx3-ubyte', label='data/train-labels-idx1-ubyte', input_shape=input_shape, batch_size=batch_size, shuffle=True, flat=flat, num_parts=num_parts, part_index=part_index)
    val_dataiter = mx.io.MNISTIter(image='data/t10k-images-idx3-ubyte', label='data/t10k-labels-idx1-ubyte', input_shape=input_shape, batch_size=batch_size, flat=flat, num_parts=num_parts, part_index=part_index)
    return (train_dataiter, val_dataiter)