import pytest
from jeepney.low_level import *
def test_array_limit():
    a = Array(FixedType(8, 'Q'))
    a.serialise(fake_list(100), 0, Endianness.little)
    with pytest.raises(SizeLimitError):
        a.serialise(fake_list(2 ** 23 + 1), 0, Endianness.little)