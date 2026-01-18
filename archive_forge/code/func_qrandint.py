import logging
from copy import copy
from inspect import signature
from math import isclose
from typing import Any, Callable, Dict, List, Optional, Sequence, Union
import numpy as np
from ray.util.annotations import DeveloperAPI, PublicAPI
@PublicAPI
def qrandint(lower: int, upper: int, q: int=1):
    """Sample an integer value uniformly between ``lower`` and ``upper``.

    ``lower`` is inclusive, ``upper`` is also inclusive (!).

    The value will be quantized, i.e. rounded to an integer increment of ``q``.
    Quantization makes the upper bound inclusive.

    .. versionchanged:: 1.5.0
        When converting Ray Tune configs to searcher-specific search spaces,
        the lower and upper limits are adjusted to keep compatibility with
        the bounds stated in the docstring above.

    """
    return Integer(lower, upper).uniform().quantized(q)