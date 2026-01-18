import yaql.tests
def test_binary_plus_int(self):
    res = self.eval('2 + 3')
    self.assertEqual(5, res)
    self.assertIsInstance(res, int)