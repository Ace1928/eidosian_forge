from __future__ import absolute_import, print_function, unicode_literals
import pytest  # NOQA
import sys
from .roundtrip import (
def test_issue_282(self):
    import srsly.ruamel_yaml
    yaml_data = srsly.ruamel_yaml.comments.CommentedMap([('a', 'apple'), ('b', 'banana')])
    yaml_data.update([('c', 'cantaloupe')])
    yaml_data.update({'d': 'date', 'k': 'kiwi'})
    assert 'c' in yaml_data.keys()
    assert 'c' in yaml_data._ok