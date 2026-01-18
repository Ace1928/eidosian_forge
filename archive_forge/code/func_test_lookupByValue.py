import warnings
from twisted.trial.unittest import TestCase
def test_lookupByValue(self):
    """
        Constants can be looked up by their associated value, defined
        implicitly by the position in which the constant appears in the class
        definition or explicitly by the argument passed to L{FlagConstant}.
        """
    flag = self.FXF.lookupByValue(1)
    self.assertIs(flag, self.FXF.READ)
    flag = self.FXF.lookupByValue(2)
    self.assertIs(flag, self.FXF.WRITE)
    flag = self.FXF.lookupByValue(4)
    self.assertIs(flag, self.FXF.APPEND)
    flag = self.FXF.lookupByValue(32)
    self.assertIs(flag, self.FXF.EXCLUSIVE)
    flag = self.FXF.lookupByValue(64)
    self.assertIs(flag, self.FXF.TEXT)