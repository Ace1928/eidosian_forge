import warnings
from twisted.trial.unittest import TestCase
def test_iterconstantsIdentity(self):
    """
        The constants returned from L{Flags.iterconstants} are identical on
        each call to that method.
        """
    constants = list(self.FXF.iterconstants())
    again = list(self.FXF.iterconstants())
    self.assertIs(again[0], constants[0])
    self.assertIs(again[1], constants[1])
    self.assertIs(again[2], constants[2])
    self.assertIs(again[3], constants[3])
    self.assertIs(again[4], constants[4])