from __future__ import print_function
import sys
import pytest  # NOQA
import platform
from .roundtrip import round_trip, dedent, round_trip_load, round_trip_dump  # NOQA
def test_CommentedSet(self):
    from srsly.ruamel_yaml.constructor import CommentedSet
    s = CommentedSet(['a', 'b', 'c'])
    s.remove('b')
    s.add('d')
    assert s == CommentedSet(['a', 'c', 'd'])
    s.add('e')
    s.add('f')
    s.remove('e')
    assert s == CommentedSet(['a', 'c', 'd', 'f'])