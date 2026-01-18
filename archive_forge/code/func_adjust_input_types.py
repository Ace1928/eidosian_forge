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
def adjust_input_types(self, indtypes):
    """
        Attempt to cast the inputs to the required types if necessary
        and if they are not device arrays.

        Side effect: Only affects the elements of `inputs` that require
        a type cast.
        """
    for i, (ity, val) in enumerate(zip(indtypes, self.inputs)):
        if ity != val.dtype:
            if not hasattr(val, 'astype'):
                msg = 'compatible signature is possible by casting but {0} does not support .astype()'.format(type(val))
                raise TypeError(msg)
            self.inputs[i] = val.astype(ity)