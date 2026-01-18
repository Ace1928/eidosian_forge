from __future__ import print_function
import pytest  # NOQA
from .roundtrip import round_trip, dedent, round_trip_load, round_trip_dump  # NOQA
def compare_eol(data, s):
    assert 'EOL' in s
    ds = dedent(s).replace('EOL', '').replace('\n', '|\n')
    assert round_trip_dump(data).replace('\n', '|\n') == ds