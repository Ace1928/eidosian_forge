import pytest
from textwrap import dedent
import platform
import srsly
from .roundtrip import (
def test_merge_accessible(self):
    from srsly.ruamel_yaml.comments import CommentedMap, merge_attrib
    data = load('\n        k: &level_2 { a: 1, b2 }\n        l: &level_1 { a: 10, c: 3 }\n        m:\n          <<: *level_1\n          c: 30\n          d: 40\n        ')
    d = data['m']
    assert isinstance(d, CommentedMap)
    assert hasattr(d, merge_attrib)