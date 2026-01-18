import yaql.tests
def test_gte(self):
    res = self.eval('5 >= 3')
    self.assertIsInstance(res, bool)
    self.assertTrue(res)
    self.assertTrue(self.eval('3 >= 3'))
    self.assertTrue(self.eval('3.5 > 3'))
    self.assertFalse(self.eval('2 >= 3'))