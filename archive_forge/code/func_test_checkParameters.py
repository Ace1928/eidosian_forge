from twisted.python import usage
from twisted.trial import unittest
def test_checkParameters(self):
    """
        Parameters have correct values.
        """
    self.assertEqual(self.nice.opts['long'], 'Alpha')
    self.assertEqual(self.nice.opts['another'], 'Beta')
    self.assertEqual(self.nice.opts['longonly'], 'noshort')
    self.assertEqual(self.nice.opts['shortless'], 'Gamma')