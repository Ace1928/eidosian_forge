import logging
from copy import copy
from inspect import signature
from math import isclose
from typing import Any, Callable, Dict, List, Optional, Sequence, Union
import numpy as np
from ray.util.annotations import DeveloperAPI, PublicAPI
@PublicAPI
def qrandn(mean: float, sd: float, q: float):
    """Sample a float value normally with ``mean`` and ``sd``.

    The value will be quantized, i.e. rounded to an integer increment of ``q``.

    Args:
        mean: Mean of the normal distribution.
        sd: SD of the normal distribution.
        q: Quantization number. The result will be rounded to an
            integer increment of this value.

    """
    return Float(None, None).normal(mean, sd).quantized(q)