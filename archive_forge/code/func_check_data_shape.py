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
def check_data_shape(self, data_shape):
    """Checks if the input data shape is valid"""
    if not len(data_shape) == 3:
        raise ValueError('data_shape should have length 3, with dimensions CxHxW')
    if not data_shape[0] == 3:
        raise ValueError('This iterator expects inputs to have 3 channels.')