import warnings
from twisted.trial.unittest import TestCase
def test_initializedOnce(self):
    """
        L{Flags._enumerants} is initialized once and its value re-used on
        subsequent access.
        """
    self._initializedOnceTest(self.FXF, 'READ')