from twisted.python import usage
from twisted.trial import unittest
def test_hasArg(self):
    """
        L{usage.flagFunction} returns C{False} if the method checked allows
        exactly one argument.
        """
    self.assertIs(False, usage.flagFunction(self.SomeClass().oneArg))