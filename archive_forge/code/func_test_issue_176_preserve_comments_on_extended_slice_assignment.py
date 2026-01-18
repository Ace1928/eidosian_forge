from __future__ import absolute_import, print_function, unicode_literals
import pytest  # NOQA
import sys
from .roundtrip import (
def test_issue_176_preserve_comments_on_extended_slice_assignment(self):
    yaml_str = dedent('        - a\n        - b  # comment\n        - c  # commment c\n        # comment c+\n        - d\n\n        - e # comment\n        ')
    seq = round_trip_load(yaml_str)
    seq[1::2] = ['B', 'D']
    res = round_trip_dump(seq)
    assert res == yaml_str.replace(' b ', ' B ').replace(' d\n', ' D\n')