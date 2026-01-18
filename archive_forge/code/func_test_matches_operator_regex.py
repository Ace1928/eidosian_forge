import yaql.tests
def test_matches_operator_regex(self):
    self.assertTrue(self.eval("axb =~ regex('a.b')"))
    self.assertFalse(self.eval("abx =~ regex('a.b')"))