import yaql.tests
def test_brackets(self):
    self.assertEqual(-4, self.eval('1 - (2) - 3'))
    self.assertEqual(2, self.eval('1 - (2 - 3)'))