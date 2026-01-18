import logging
from copy import copy
from inspect import signature
from math import isclose
from typing import Any, Callable, Dict, List, Optional, Sequence, Union
import numpy as np
from ray.util.annotations import DeveloperAPI, PublicAPI
@PublicAPI
def qlograndint(lower: int, upper: int, q: int, base: float=10):
    """Sample an integer value log-uniformly between ``lower`` and ``upper``,
    with ``base`` being the base of logarithm.

    ``lower`` is inclusive, ``upper`` is also inclusive (!).

    The value will be quantized, i.e. rounded to an integer increment of ``q``.
    Quantization makes the upper bound inclusive.

    .. versionchanged:: 1.5.0
        When converting Ray Tune configs to searcher-specific search spaces,
        the lower and upper limits are adjusted to keep compatibility with
        the bounds stated in the docstring above.

    """
    return Integer(lower, upper).loguniform(base).quantized(q)