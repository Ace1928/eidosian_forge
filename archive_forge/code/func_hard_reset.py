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
def hard_reset(self):
    """Resets the iterator and ignore roll over data"""
    if self.seq is not None and self.shuffle:
        random.shuffle(self.seq)
    if self.imgrec is not None:
        self.imgrec.reset()
    self.cur = 0
    self._allow_read = True
    self._cache_data = None
    self._cache_label = None
    self._cache_idx = None