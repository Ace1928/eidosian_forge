import contextlib
from functools import partial
from unittest import TestCase
from unittest.util import safe_repr
import numpy as np
import pandas as pd
from numpy.testing import assert_array_almost_equal, assert_array_equal
from ..core import (
from ..core.options import Cycle, Options
from ..core.util import cast_array_to_int64, datetime_types, dt_to_int, is_float
from . import *  # noqa (All Elements need to support comparison)
@classmethod
def compare_dimensions(cls, dim1, dim2, msg=None):
    if dim1.name != dim2.name:
        raise cls.failureException(f'Dimension names mismatched: {dim1.name} != {dim2.name}')
    if dim1.label != dim2.label:
        raise cls.failureException(f'Dimension labels mismatched: {dim1.label} != {dim2.label}')
    dim1_params = dim1.param.values()
    dim2_params = dim2.param.values()
    if set(dim1_params.keys()) != set(dim2_params.keys()):
        raise cls.failureException(f'Dimension parameter sets mismatched: {set(dim1_params.keys())} != {set(dim2_params.keys())}')
    for k in dim1_params.keys():
        if dim1.param.objects('existing')[k].__class__.__name__ == 'Callable' and dim2.param.objects('existing')[k].__class__.__name__ == 'Callable':
            continue
        try:
            cls.assertEqual(dim1_params[k], dim2_params[k], msg=None)
        except AssertionError as e:
            msg = f'Dimension parameter {k!r} mismatched: '
            raise cls.failureException(f'{msg}{e!s}') from e