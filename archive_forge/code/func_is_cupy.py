from __future__ import annotations
import os
import warnings
import numpy as np
from numpy.testing import assert_
import scipy._lib.array_api_compat.array_api_compat as array_api_compat
from scipy._lib.array_api_compat.array_api_compat import size
import scipy._lib.array_api_compat.array_api_compat.numpy as array_api_compat_numpy
def is_cupy(xp):
    return xp.__name__ == 'scipy._lib.array_api_compat.array_api_compat.cupy'