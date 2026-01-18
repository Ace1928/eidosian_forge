from collections import namedtuple
import datetime
import decimal
from functools import lru_cache, partial
import inspect
import itertools
import math
import os
import pytest
import random
import sys
import textwrap
import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
from pyarrow.lib import ArrowNotImplementedError
from pyarrow.tests import util
def largest_scaled_float_not_above(val, scale):
    """
    Find the largest float f such as `f * 10**scale <= val`
    """
    assert val >= 0
    assert scale >= 0
    float_val = float(val) / 10 ** scale
    if float_val * 10 ** scale > val:
        float_val = np.nextafter(float_val, 0.0)
        if float_val * 10 ** scale > val:
            float_val = np.nextafter(float_val, 0.0)
    assert float_val * 10 ** scale <= val
    return float_val