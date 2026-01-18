from __future__ import print_function
import pytest  # NOQA
from .roundtrip import round_trip, dedent, round_trip_load, round_trip_dump  # NOQA
def test_map_set_comment_on_existing_column_further_away(self):
    """
        no comment line before or after, take the latest before
        the new position
        """
    data = load('\n            a: 1   # comment 1\n            b: 2\n            c: 3\n            d: 4\n            e: 5     # comment 3\n            ')
    data.yaml_add_eol_comment('comment 2', key='c')
    print(round_trip_dump(data))
    exp = '\n            a: 1   # comment 1\n            b: 2\n            c: 3   # comment 2\n            d: 4\n            e: 5     # comment 3\n            '
    compare(data, exp)