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
def check_cast_float_to_decimal(float_ty, float_val, decimal_ty, decimal_ctx, max_precision):
    decimal_ctx.prec = decimal_ty.precision
    decimal_ctx.rounding = decimal.ROUND_HALF_EVEN
    expected = decimal_ctx.create_decimal_from_float(float_val)
    expected = expected.quantize(decimal.Decimal(1).scaleb(-decimal_ty.scale))
    s = pa.scalar(float_val, type=float_ty)
    actual = pc.cast(s, decimal_ty).as_py()
    if actual != expected:
        diff_digits = abs(actual - expected) * 10 ** decimal_ty.scale
        limit = 2 if decimal_ty.precision < max_precision - 1 else 4
        assert diff_digits <= limit, f'float_val = {float_val!r}, precision={decimal_ty.precision}, expected = {expected!r}, actual = {actual!r}, diff_digits = {diff_digits!r}'