import pytest
from textwrap import dedent
import platform
import srsly
from .roundtrip import (
def test_issue_196_cast_of_dict(self, capsys):
    from srsly.ruamel_yaml import YAML
    yaml = YAML()
    mapping = yaml.load('        anchored: &anchor\n          a : 1\n\n        mapping:\n          <<: *anchor\n          b: 2\n        ')['mapping']
    for k in mapping:
        print('k', k)
    for k in mapping.copy():
        print('kc', k)
    print('v', list(mapping.keys()))
    print('v', list(mapping.values()))
    print('v', list(mapping.items()))
    print(len(mapping))
    print('-----')
    assert 'a' in mapping
    x = {}
    for k in mapping:
        x[k] = mapping[k]
    assert 'a' in x
    assert 'a' in mapping.keys()
    assert mapping['a'] == 1
    assert mapping.__getitem__('a') == 1
    assert 'a' in dict(mapping)
    assert 'a' in dict(mapping.items())