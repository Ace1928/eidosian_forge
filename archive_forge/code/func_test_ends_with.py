from yaql.language import exceptions
import yaql.tests
def test_ends_with(self):
    self.assertTrue(self.eval('ABC.endsWith(C)'))
    self.assertTrue(self.eval('ABC.endsWith(B, C)'))
    self.assertFalse(self.eval('ABC.endsWith(B)'))
    self.assertRaises(exceptions.NoMatchingMethodException, self.eval, 'ABC.endsWith(null)')