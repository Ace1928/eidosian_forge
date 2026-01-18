from __future__ import absolute_import, print_function, unicode_literals
import pytest  # NOQA
import sys
from .roundtrip import (
def test_issue_295(self):
    import copy
    inp = dedent("\n        A:\n          b:\n          # comment\n          - l1\n          - l2\n\n        C:\n          d: e\n          f:\n          # comment2\n          - - l31\n            - l32\n            - l33: '5'\n        ")
    data = round_trip_load(inp)
    dc = copy.deepcopy(data)
    assert round_trip_dump(dc) == inp