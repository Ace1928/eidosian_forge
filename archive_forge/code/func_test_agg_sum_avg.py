from qpd_test.tests_base import TestsBase
import pandas as pd
def test_agg_sum_avg(self):
    a = self.make_rand_df(100, a=(int, 50), b=(str, 50), c=(int, 30), d=(str, 40), e=(float, 40))
    self.eq_sqlite('\n                SELECT\n                    a,b, a+1 AS c,\n                    SUM(c) AS sum_c,\n                    AVG(c) AS avg_c,\n                    SUM(e) AS sum_e,\n                    AVG(e) AS avg_e,\n                    SUM(a)+AVG(e) AS mix_1,\n                    SUM(a+e) AS mix_2\n                FROM a GROUP BY a,b\n                ', a=a)