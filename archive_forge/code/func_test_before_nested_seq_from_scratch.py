from __future__ import print_function
import pytest  # NOQA
from .roundtrip import round_trip, dedent, round_trip_load, round_trip_dump  # NOQA
def test_before_nested_seq_from_scratch(self):
    from srsly.ruamel_yaml.comments import CommentedMap, CommentedSeq
    data = CommentedMap()
    datab = CommentedSeq()
    data['a'] = 1
    data['b'] = datab
    datab.append('c')
    datab.append('d')
    data['b'].yaml_set_start_comment('Hello\nWorld\n', indent=2)
    exp = '\n        a: 1\n        b:\n          # Hello\n          # World\n        - c\n        - d\n        '
    compare(data, exp.format(comment='#'))