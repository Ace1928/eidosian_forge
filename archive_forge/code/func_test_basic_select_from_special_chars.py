from qpd_test.tests_base import TestsBase
import pandas as pd
def test_basic_select_from_special_chars(self):
    df = self.make_rand_df(5, **{'a b': (int, 2), '-': (str, 3), 'c': (float, 4)})
    self.eq_sqlite("SELECT 1 AS `a c`, 1.5 AS `-`, 'x' AS c")
    self.eq_sqlite("SELECT 1+2 AS `a b`, 1.5*3 AS b, 'x' AS c")
    self.eq_sqlite('SELECT *, 1 AS `c d` FROM a', a=df)
    self.eq_sqlite('SELECT * FROM a AS x', a=df)
    self.eq_sqlite('SELECT `-` AS `b b `, `a b`+1-2*3.0/4 AS `cc`, x.* FROM a AS x', a=df)
    self.eq_sqlite("SELECT *, 1 AS x, 2.5 AS y, 'z' AS z FROM a AS x", a=df)
    self.eq_sqlite('SELECT *, -(1.0+`a b`)/3 AS x, +(2.5) AS y FROM a AS x', a=df)