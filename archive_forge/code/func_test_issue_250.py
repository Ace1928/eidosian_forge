from __future__ import absolute_import, print_function, unicode_literals
import pytest  # NOQA
import sys
from .roundtrip import (
def test_issue_250(self):
    inp = '\n        # 1.\n        - - 1\n        # 2.\n        - map: 2\n        # 3.\n        - 4\n        '
    d = round_trip(inp)