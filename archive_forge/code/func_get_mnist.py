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
def get_mnist():
    """Download and load the MNIST dataset

    Returns
    -------
    dict
        A dict containing the data
    """

    def read_data(label_url, image_url):
        with gzip.open(mx.test_utils.download(label_url)) as flbl:
            struct.unpack('>II', flbl.read(8))
            label = np.frombuffer(flbl.read(), dtype=np.int8)
        with gzip.open(mx.test_utils.download(image_url), 'rb') as fimg:
            _, _, rows, cols = struct.unpack('>IIII', fimg.read(16))
            image = np.frombuffer(fimg.read(), dtype=np.uint8).reshape(len(label), rows, cols)
            image = image.reshape(image.shape[0], 1, 28, 28).astype(np.float32) / 255
        return (label, image)
    path = 'http://data.mxnet.io/data/mnist/'
    train_lbl, train_img = read_data(path + 'train-labels-idx1-ubyte.gz', path + 'train-images-idx3-ubyte.gz')
    test_lbl, test_img = read_data(path + 't10k-labels-idx1-ubyte.gz', path + 't10k-images-idx3-ubyte.gz')
    return {'train_data': train_img, 'train_label': train_lbl, 'test_data': test_img, 'test_label': test_lbl}