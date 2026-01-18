import os
import sys
import hashlib
import uuid
import warnings
import collections
import weakref
import requests
import numpy as np
from .. import ndarray
from ..util import is_np_shape, is_np_array
from .. import numpy as _mx_np  # pylint: disable=reimported
def shape_is_known(shape):
    """Check whether a shape is completely known with or without np semantics.

    Please see the doc of is_np_shape for more details.
    """
    if shape is None:
        return False
    unknown_dim_size = -1 if is_np_shape() else 0
    if len(shape) == 0:
        return unknown_dim_size == -1
    for dim_size in shape:
        if dim_size == unknown_dim_size:
            return False
        assert dim_size > unknown_dim_size, 'shape dimension size cannot be less than {}, while received {}'.format(unknown_dim_size, dim_size)
    return True