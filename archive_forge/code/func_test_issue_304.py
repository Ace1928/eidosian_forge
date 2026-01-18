from __future__ import absolute_import, print_function, unicode_literals
import pytest  # NOQA
import sys
from .roundtrip import (
def test_issue_304(self):
    inp = '\n        %YAML 1.2\n        %TAG ! tag:example.com,2019:\n        ---\n        !foo null\n        ...\n        '
    d = na_round_trip(inp)