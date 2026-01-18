from __future__ import absolute_import, print_function, unicode_literals
import pytest  # NOQA
import sys
from .roundtrip import (
def test_issue_172(self):
    x = round_trip_load(TestIssues.json_str2)
    x = round_trip_load(TestIssues.json_str)