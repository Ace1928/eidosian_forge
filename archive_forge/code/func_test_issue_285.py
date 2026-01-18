from __future__ import absolute_import, print_function, unicode_literals
import pytest  # NOQA
import sys
from .roundtrip import (
def test_issue_285(self):
    from srsly.ruamel_yaml import YAML
    yaml = YAML()
    inp = dedent('        %YAML 1.1\n        ---\n        - y\n        - n\n        - Y\n        - N\n        ')
    a = yaml.load(inp)
    assert a[0]
    assert a[2]
    assert not a[1]
    assert not a[3]