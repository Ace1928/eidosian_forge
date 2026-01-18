from qpd_test.tests_base import TestsBase
import pandas as pd
def test_in_between(self):
    df = self.make_rand_df(10, a=(int, 3), b=(str, 3))
    self.eq_sqlite('SELECT * FROM a WHERE a IN (2,4,6)', a=df)
    self.eq_sqlite('SELECT * FROM a WHERE a BETWEEN 2 AND 4+1', a=df)
    self.eq_sqlite('SELECT * FROM a WHERE a NOT IN (2,4,6)', a=df)
    self.eq_sqlite('SELECT * FROM a WHERE a NOT BETWEEN 2 AND 4+1', a=df)