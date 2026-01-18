import datetime
import decimal
from twisted.internet.testing import StringTransport
from twisted.spread import banana, jelly, pb
from twisted.trial import unittest
from twisted.trial.unittest import TestCase
def test_newStyleWithSlots(self):
    """
        A class defined with I{slots} can be jellied and unjellied with the
        values for its attributes preserved.
        """
    n = E()
    n.x = 1
    c = jelly.jelly(n)
    m = jelly.unjelly(c)
    self.assertIsInstance(m, E)
    self.assertEqual(n.x, 1)