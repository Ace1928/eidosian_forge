import sys
import os
from six import iteritems
from enum import IntEnum
from contextlib import contextmanager
import json
@property
def n_samples_fit_(self):
    """
        Returns
        -------
        Number of samples in the fitted data.
        """
    self._check_index()
    return self._index_data.shape[0]