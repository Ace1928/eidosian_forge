from __future__ import absolute_import, print_function, unicode_literals
import pytest  # NOQA
import sys
from .roundtrip import (
def test_issue_249(self):
    yaml = YAML()
    inp = dedent('        # comment\n        -\n          - 1\n          - 2\n          - 3\n        ')
    exp = dedent('        # comment\n        - - 1\n          - 2\n          - 3\n        ')
    yaml.round_trip(inp, outp=exp)