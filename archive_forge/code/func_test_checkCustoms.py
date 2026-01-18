from twisted.python import usage
from twisted.trial import unittest
def test_checkCustoms(self):
    """
        Custom flags and parameters have correct values.
        """
    self.assertEqual(self.nice.opts['myflag'], 'PONY!')
    self.assertEqual(self.nice.opts['myparam'], 'Tofu WITH A PONY!')