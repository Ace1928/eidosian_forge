import yaql.tests
def test_matches_operator_string(self):
    self.assertTrue(self.eval("axb =~ 'a.b'"))
    self.assertFalse(self.eval("abx =~ 'a.b'"))