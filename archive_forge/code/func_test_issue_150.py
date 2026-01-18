from __future__ import absolute_import, print_function, unicode_literals
import pytest  # NOQA
import sys
from .roundtrip import (
def test_issue_150(self):
    from srsly.ruamel_yaml import YAML
    inp = '        base: &base_key\n          first: 123\n          second: 234\n\n        child:\n          <<: *base_key\n          third: 345\n        '
    yaml = YAML()
    data = yaml.load(inp)
    child = data['child']
    assert 'second' in dict(**child)