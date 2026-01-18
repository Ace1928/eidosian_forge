from __future__ import absolute_import, print_function, unicode_literals
import pytest  # NOQA
import sys
from .roundtrip import (
def test_issue_161(self):
    yaml_str = dedent('        mapping-A:\n          key-A:{}\n        mapping-B:\n        ')
    for comment in ['', ' # no-newline', '  # some comment\n', '\n']:
        s = yaml_str.format(comment)
        res = round_trip(s)