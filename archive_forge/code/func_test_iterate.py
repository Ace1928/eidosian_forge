import warnings
from twisted.trial.unittest import TestCase
def test_iterate(self):
    """
        A L{FlagConstant} instance which results from C{|} can be
        iterated upon to yield the original constants.
        """
    self.assertEqual(set(self.FXF.WRITE & self.FXF.READ), set())
    self.assertEqual(set(self.FXF.WRITE), {self.FXF.WRITE})
    self.assertEqual(set(self.FXF.WRITE | self.FXF.EXCLUSIVE), {self.FXF.WRITE, self.FXF.EXCLUSIVE})