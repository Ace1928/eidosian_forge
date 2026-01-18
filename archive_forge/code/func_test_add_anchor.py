import pytest
from textwrap import dedent
import platform
import srsly
from .roundtrip import (
def test_add_anchor(self):
    from srsly.ruamel_yaml.comments import CommentedMap
    data = CommentedMap()
    data_a = CommentedMap()
    data['a'] = data_a
    data_a['c'] = 3
    data['b'] = 2
    data.yaml_set_anchor('klm', always_dump=True)
    data['a'].yaml_set_anchor('xyz', always_dump=True)
    compare(data, '\n        &klm\n        a: &xyz\n          c: 3\n        b: 2\n        ')