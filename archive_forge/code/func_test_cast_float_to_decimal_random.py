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
@pytest.mark.parametrize('float_ty', [pa.float32(), pa.float64()], ids=str)
@pytest.mark.parametrize('decimal_traits', decimal_type_traits, ids=lambda v: v.name)
def test_cast_float_to_decimal_random(float_ty, decimal_traits):
    """
    Test float-to-decimal conversion against exactly generated values.
    """
    r = random.Random(43)
    np_float_ty = {pa.float32(): np.float32, pa.float64(): np.float64}[float_ty]
    mantissa_bits = {pa.float32(): 24, pa.float64(): 53}[float_ty]
    float_exp_min, float_exp_max = {pa.float32(): (-126, 127), pa.float64(): (-1022, 1023)}[float_ty]
    mantissa_digits = math.floor(math.log10(2 ** mantissa_bits))
    max_precision = decimal_traits.max_precision
    with decimal.localcontext() as ctx:
        precision = mantissa_digits
        ctx.prec = precision
        min_scale = max(-max_precision, precision + math.ceil(math.log10(2 ** float_exp_min)))
        max_scale = min(max_precision, math.floor(math.log10(2 ** float_exp_max)))
        for scale in range(min_scale, max_scale):
            decimal_ty = decimal_traits.factory(precision, scale)
            float_exp = -mantissa_bits + math.floor(math.log2(10 ** (precision - scale)))
            assert float_exp_min <= float_exp <= float_exp_max
            for i in range(5):
                mantissa = r.randrange(0, 2 ** mantissa_bits)
                float_val = np.ldexp(np_float_ty(mantissa), float_exp)
                assert isinstance(float_val, np_float_ty)
                if float_exp >= 0:
                    expected = decimal.Decimal(mantissa) * 2 ** float_exp
                else:
                    expected = decimal.Decimal(mantissa) / 2 ** (-float_exp)
                expected_as_int = round(expected.scaleb(scale))
                actual = pc.cast(pa.scalar(float_val, type=float_ty), decimal_ty).as_py()
                actual_as_int = round(actual.scaleb(scale))
                assert abs(actual_as_int - expected_as_int) <= 1