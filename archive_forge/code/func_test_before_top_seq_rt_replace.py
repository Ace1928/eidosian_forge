from __future__ import print_function
import pytest  # NOQA
from .roundtrip import round_trip, dedent, round_trip_load, round_trip_dump  # NOQA
def test_before_top_seq_rt_replace(self):
    s = '\n        # this\n        # that\n        - a\n        - b\n        '
    data = load(s.format(comment='#'))
    data.yaml_set_start_comment('Hello\nWorld\n')
    print(round_trip_dump(data))
    exp = '\n        # Hello\n        # World\n        - a\n        - b\n        '
    compare(data, exp.format(comment='#'))