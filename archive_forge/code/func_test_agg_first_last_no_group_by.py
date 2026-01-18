from qpd_test.tests_base import TestsBase
import pandas as pd
def test_agg_first_last_no_group_by(self):
    a = ([[1, 'x', None], [2, None, 2.5], [2, None, 2.5]], ['a', 'b', 'c'])
    self.assert_eq(dict(a=a), '\n                SELECT\n                    FIRST(a) AS a1,\n                    LAST(a) AS a2,\n                    FIRST_VALUE(b) AS a3,\n                    LAST_VALUE(b) AS a4,\n                    FIRST_VALUE(c) AS a5,\n                    LAST_VALUE(c) AS a6\n                FROM a\n                ', [[1, 2, 'x', None, None, 2.5]], ['a1', 'a2', 'a3', 'a4', 'a5', 'a6'])