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
def postprocess_data(self, datum):
    """Final postprocessing step before image is loaded into the batch."""
    if is_np_array():
        return datum.transpose(2, 0, 1)
    else:
        return nd.transpose(datum, axes=(2, 0, 1))