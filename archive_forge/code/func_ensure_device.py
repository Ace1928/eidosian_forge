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
def ensure_device(parameter):
    if self.is_device_array(parameter):
        convert = self.as_device_array
    else:
        convert = self.to_device
    return convert(parameter)