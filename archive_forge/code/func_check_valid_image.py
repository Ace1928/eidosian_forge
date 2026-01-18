import sys
import os
import random
import logging
import json
import warnings
from numbers import Number
import numpy as np
from .. import numpy as _mx_np  # pylint: disable=reimported
from ..base import numeric_types
from .. import ndarray as nd
from ..ndarray import _internal
from .. import io
from .. import recordio
from .. util import is_np_array
from ..ndarray.numpy import _internal as _npi
def check_valid_image(self, data):
    """Checks if the input data is valid"""
    if len(data[0].shape) == 0:
        raise RuntimeError('Data shape is wrong')