import logging
import math
import pickle
import warnings
import os
import numpy
from ..base import py_str
from ..ndarray import (NDArray, zeros, clip, sqrt, cast, maximum, abs as NDabs, array, multiply,
from ..ndarray import (sgd_update, sgd_mom_update, adam_update, rmsprop_update, rmspropalex_update,
from ..ndarray.contrib import (multi_lamb_update, multi_mp_lamb_update)
from ..ndarray import sparse
from ..random import normal
from ..util import is_np_array
def sync_state_context(self, state, context):
    """sync state context."""
    if isinstance(state, NDArray):
        return state.as_in_context(context)
    elif isinstance(state, (tuple, list)):
        synced_state = (self.sync_state_context(i, context) for i in state)
        if isinstance(state, tuple):
            return tuple(synced_state)
        else:
            return list(synced_state)
    else:
        return state