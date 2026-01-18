from qpd_test.tests_base import TestsBase
import pandas as pd
def test_agg_min_max_no_group_by(self):
    a = self.make_rand_df(100, a=(int, 50), b=(str, 50), c=(int, 30), d=(str, 40), e=(float, 40))
    self.eq_sqlite('\n                SELECT\n                    MIN(a) AS min_a,\n                    MAX(a) AS max_a,\n                    MIN(b) AS min_b,\n                    MAX(b) AS max_b,\n                    MIN(c) AS min_c,\n                    MAX(c) AS max_c,\n                    MIN(d) AS min_d,\n                    MAX(d) AS max_d,\n                    MIN(e) AS min_e,\n                    MAX(e) AS max_e,\n                    MIN(a+e) AS mix_1,\n                    MIN(a)+MIN(e) AS mix_2\n                FROM a\n                ', a=a)