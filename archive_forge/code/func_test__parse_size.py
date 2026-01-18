import sys
from scipy._lib._testutils import _parse_size, _get_mem_available
import pytest
def test__parse_size():
    expected = {'12': 12000000.0, '12 b': 12, '12k': 12000.0, '  12  M  ': 12000000.0, '  12  G  ': 12000000000.0, ' 12Tb ': 12000000000000.0, '12  Mib ': 12 * 1024.0 ** 2, '12Tib': 12 * 1024.0 ** 4}
    for inp, outp in sorted(expected.items()):
        if outp is None:
            with pytest.raises(ValueError):
                _parse_size(inp)
        else:
            assert _parse_size(inp) == outp