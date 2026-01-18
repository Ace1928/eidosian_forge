from __future__ import annotations
from typing import TYPE_CHECKING, Callable, Any, cast
import numpy as np
import numpy.typing as npt
import math
import warnings
from collections import namedtuple
from scipy.special import roots_legendre
from scipy.special import gammaln, logsumexp
from scipy._lib._util import _rng_spawn
from scipy._lib.deprecation import (_NoValue, _deprecate_positional_args,
def cumtrapz(y, x=None, dx=1.0, axis=-1, initial=None):
    """An alias of `cumulative_trapezoid`.

    `cumtrapz` is kept for backwards compatibility. For new code, prefer
    `cumulative_trapezoid` instead.
    """
    msg = "'scipy.integrate.cumtrapz' is deprecated in favour of 'scipy.integrate.cumulative_trapezoid' and will be removed in SciPy 1.14.0"
    warnings.warn(msg, DeprecationWarning, stacklevel=2)
    return cumulative_trapezoid(y, x=x, dx=dx, axis=axis, initial=initial)