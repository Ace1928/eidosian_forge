import yaql.tests
def test_comparision_of_incomparable(self):
    self.assertFalse(self.eval('a = 1'))
    self.assertFalse(self.eval('a = false'))
    self.assertFalse(self.eval('a = null'))
    self.assertFalse(self.eval('[a] = [false]'))
    self.assertTrue(self.eval('a != 1'))
    self.assertTrue(self.eval('a != false'))
    self.assertTrue(self.eval('[a] != [false]'))
    self.assertTrue(self.eval('a != null'))