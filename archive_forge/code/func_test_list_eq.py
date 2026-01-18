from yaql.language import exceptions
import yaql.tests
def test_list_eq(self):
    self.assertTrue(self.eval('[c, 55]=[c, 55]'))
    self.assertFalse(self.eval('[c, 55]=[55, c]'))
    self.assertFalse(self.eval('[c, 55]=null'))
    self.assertFalse(self.eval('null = [c, 55]'))