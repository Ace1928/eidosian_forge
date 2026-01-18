import pytest
import sys
from .roundtrip import round_trip, dedent, round_trip_load, round_trip_dump
def test_mlget_00(self):
    x = '        a:\n        - b:\n          c: 42\n        - d:\n            f: 196\n          e:\n            g: 3.14\n        '
    d = round_trip_load(x)
    assert d.mlget(['a', 1, 'd', 'f'], list_ok=True) == 196
    with pytest.raises(AssertionError):
        d.mlget(['a', 1, 'd', 'f']) == 196