import warnings
from twisted.trial.unittest import TestCase
def test_lookupByName(self):
    """
        Constants can be looked up by name using L{Flags.lookupByName}.
        """
    flag = self.FXF.lookupByName('READ')
    self.assertIs(self.FXF.READ, flag)