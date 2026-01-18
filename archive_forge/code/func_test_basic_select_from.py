from qpd_test.tests_base import TestsBase
import pandas as pd
def test_basic_select_from(self):
    df = self.make_rand_df(5, a=(int, 2), b=(str, 3), c=(float, 4))
    self.eq_sqlite("SELECT 1 AS a, 1.5 AS b, 'x' AS c")
    self.eq_sqlite("SELECT 1+2 AS a, 1.5*3 AS b, 'x' AS c")
    self.eq_sqlite('SELECT * FROM a', a=df)
    self.eq_sqlite('SELECT * FROM a AS x', a=df)
    self.eq_sqlite('SELECT b AS bb, a+1-2*3.0/4 AS cc, x.* FROM a AS x', a=df)
    self.eq_sqlite("SELECT *, 1 AS x, 2.5 AS y, 'z' AS z FROM a AS x", a=df)
    self.eq_sqlite('SELECT *, -(1.0+a)/3 AS x, +(2.5) AS y FROM a AS x', a=df)