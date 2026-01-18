from abc import ABCMeta, abstractmethod
from collections import OrderedDict
import operator
import warnings
from functools import reduce
import numpy as np
from numba.np.ufunc.ufuncbuilder import _BaseUFuncBuilder, parse_identity
from numba.core import types, sigutils
from numba.core.typing import signature
from numba.np.ufunc.sigparse import parse_signature
def broadcast_device(self, ary, shape):
    """Handles ondevice broadcasting

        Override in subclass to add support.
        """
    raise NotImplementedError('broadcasting on device is not supported')