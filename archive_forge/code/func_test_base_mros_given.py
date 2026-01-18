import unittest
def test_base_mros_given(self):
    c3 = self._makeOne(type(self), base_mros={unittest.TestCase: unittest.TestCase.__mro__})
    memo = c3.memo
    self.assertIn(unittest.TestCase, memo)
    self.assertIsNone(memo[unittest.TestCase].had_inconsistency)