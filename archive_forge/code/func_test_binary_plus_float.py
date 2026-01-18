import yaql.tests
def test_binary_plus_float(self):
    res = self.eval('2 + 3.0')
    self.assertEqual(5, res)
    self.assertIsInstance(res, float)
    res = self.eval('2.3+3.5')
    self.assertEqual(5.8, res)
    self.assertIsInstance(res, float)