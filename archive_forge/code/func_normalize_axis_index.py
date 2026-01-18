import re
from contextlib import contextmanager
import functools
import operator
import warnings
import numbers
from collections import namedtuple
import inspect
import math
from typing import (
import numpy as np
from scipy._lib._array_api import array_namespace
def normalize_axis_index(axis, ndim):
    if axis < -ndim or axis >= ndim:
        msg = f'axis {axis} is out of bounds for array of dimension {ndim}'
        raise AxisError(msg)
    if axis < 0:
        axis = axis + ndim
    return axis