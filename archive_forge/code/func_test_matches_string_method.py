import yaql.tests
def test_matches_string_method(self):
    self.assertTrue(self.eval("axb.matches('a.b')"))
    self.assertFalse(self.eval("abx.matches('a.b')"))