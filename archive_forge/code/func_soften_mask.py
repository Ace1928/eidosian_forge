import builtins
import inspect
import operator
import warnings
import textwrap
import re
from functools import reduce
import numpy as np
import numpy.core.umath as umath
import numpy.core.numerictypes as ntypes
from numpy.core import multiarray as mu
from numpy import ndarray, amax, amin, iscomplexobj, bool_, _NoValue
from numpy import array as narray
from numpy.lib.function_base import angle
from numpy.compat import (
from numpy import expand_dims
from numpy.core.numeric import normalize_axis_tuple
frombuffer = _convert2ma(
fromfunction = _convert2ma(
def soften_mask(self):
    """
        Force the mask to soft (default), allowing unmasking by assignment.

        Whether the mask of a masked array is hard or soft is determined by
        its `~ma.MaskedArray.hardmask` property. `soften_mask` sets
        `~ma.MaskedArray.hardmask` to ``False`` (and returns the modified
        self).

        See Also
        --------
        ma.MaskedArray.hardmask
        ma.MaskedArray.harden_mask

        """
    self._hardmask = False
    return self