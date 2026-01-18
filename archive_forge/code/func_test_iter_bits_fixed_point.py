import math
import random
import pytest
from cirq_ft.infra.bit_tools import (
@pytest.mark.parametrize('val', [random.uniform(-1, 1) for _ in range(10)])
@pytest.mark.parametrize('width', [*range(2, 20, 2)])
@pytest.mark.parametrize('signed', [True, False])
def test_iter_bits_fixed_point(val, width, signed):
    if val < 0 and (not signed):
        with pytest.raises(AssertionError):
            _ = [*iter_bits_fixed_point(val, width, signed=signed)]
    else:
        bits = [*iter_bits_fixed_point(val, width, signed=signed)]
        if signed:
            sign, bits = (bits[0], bits[1:])
            assert sign == (1 if val < 0 else 0)
        val = abs(val)
        approx_val = math.fsum([b * (1 / 2 ** (1 + i)) for i, b in enumerate(bits)])
        unsigned_width = width - 1 if signed else width
        assert math.isclose(val, approx_val, abs_tol=1 / 2 ** unsigned_width), f'{val}:{approx_val}:{width}'
        bits_from_int = [*iter_bits(float_as_fixed_width_int(val, unsigned_width + 1)[1], unsigned_width)]
        assert bits == bits_from_int