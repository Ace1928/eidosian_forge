from twisted.python import usage
from twisted.trial import unittest
def test_checkFlags(self):
    """
        Flags have correct values.
        """
    self.assertEqual(self.nice.opts['aflag'], 1)
    self.assertEqual(self.nice.opts['flout'], 0)