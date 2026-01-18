import yaql.tests
def test_modulo_int(self):
    res = self.eval('9 mod 5')
    self.assertEqual(4, res)
    self.assertIsInstance(res, int)
    self.assertEqual(-1, self.eval('9 mod -5'))