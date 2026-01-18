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
def imdecode(self, s):
    """Decodes a string or byte string to an NDArray.
        See mx.img.imdecode for more details."""

    def locate():
        """Locate the image file/index if decode fails."""
        if self.seq is not None:
            idx = self.seq[self.cur % self.num_image - 1]
        else:
            idx = self.cur % self.num_image - 1
        if self.imglist is not None:
            _, fname = self.imglist[idx]
            msg = 'filename: {}'.format(fname)
        else:
            msg = 'index: {}'.format(idx)
        return 'Broken image ' + msg
    try:
        img = imdecode(s)
    except Exception as e:
        raise RuntimeError('{}, {}'.format(locate(), e))
    return img