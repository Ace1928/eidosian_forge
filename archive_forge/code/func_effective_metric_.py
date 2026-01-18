import sys
import os
from six import iteritems
from enum import IntEnum
from contextlib import contextmanager
import json
@property
def effective_metric_(self):
    """
        Returns
        -------
        Distance that should be used for finding nearest vectors.
        """
    return self.distance