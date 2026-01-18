from __future__ import absolute_import
from __future__ import print_function
from __future__ import unicode_literals
import pytest  # NOQA
from .roundtrip import round_trip, round_trip_load, round_trip_dump, dedent, YAML
def test_guess_20(self):
    inp = '        a:\n        - 1\n        '
    assert guess(inp) == (2, 0)