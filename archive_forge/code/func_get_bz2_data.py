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
def get_bz2_data(data_dir, data_name, url, data_origin_name):
    """Download and extract bz2 data.

    Parameters
    ----------

    data_dir : str
        Absolute or relative path of the directory name to store bz2 files
    data_name : str
        Name of the output file in which bz2 contents will be extracted
    url : str
        URL to download data from
    data_origin_name : str
        Name of the downloaded b2 file

    Examples
    --------
    >>> get_bz2_data("data_dir", "kdda.t",
                     "https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/binary/kdda.t.bz2",
                     "kdda.t.bz2")
    """
    data_name = os.path.join(data_dir, data_name)
    data_origin_name = os.path.join(data_dir, data_origin_name)
    if not os.path.exists(data_name):
        download(url, fname=data_origin_name, dirname=data_dir, overwrite=False)
        bz_file = bz2.BZ2File(data_origin_name, 'rb')
        with open(data_name, 'wb') as fout:
            for line in bz_file:
                fout.write(line)
            bz_file.close()
        os.remove(data_origin_name)