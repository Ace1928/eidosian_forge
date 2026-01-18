from __future__ import print_function
import pytest  # NOQA
from .roundtrip import round_trip, dedent, round_trip_load, round_trip_dump  # NOQA
def test_before_top_map_from_scratch(self):
    from srsly.ruamel_yaml.comments import CommentedMap
    data = CommentedMap()
    data['a'] = 1
    data['b'] = 2
    data.yaml_set_start_comment('Hello\nWorld\n')
    exp = '\n            # Hello\n            # World\n            a: 1\n            b: 2\n            '
    compare(data, exp.format(comment='#'))