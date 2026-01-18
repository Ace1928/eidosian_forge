import pytest
from textwrap import dedent
import platform
import srsly
from .roundtrip import (
def test_issue_213_copy_of_merge(self):
    from srsly.ruamel_yaml import YAML
    yaml = YAML()
    d = yaml.load('        foo: &foo\n          a: a\n        foo2:\n          <<: *foo\n          b: b\n        ')['foo2']
    assert d['a'] == 'a'
    d2 = d.copy()
    assert d2['a'] == 'a'
    print('d', d)
    del d['a']
    assert 'a' not in d
    assert 'a' in d2