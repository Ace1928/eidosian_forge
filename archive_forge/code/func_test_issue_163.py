from __future__ import absolute_import, print_function, unicode_literals
import pytest  # NOQA
import sys
from .roundtrip import (
def test_issue_163(self):
    s = dedent('        some-list:\n        # List comment\n        - {}\n        ')
    x = round_trip(s, preserve_quotes=True)