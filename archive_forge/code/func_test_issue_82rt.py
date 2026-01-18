from __future__ import absolute_import, print_function, unicode_literals
import pytest  # NOQA
import sys
from .roundtrip import (
def test_issue_82rt(self, tmpdir):
    yaml_str = '[1, 2, 3, !si 10k, 100G]\n'
    x = round_trip(yaml_str, preserve_quotes=True)