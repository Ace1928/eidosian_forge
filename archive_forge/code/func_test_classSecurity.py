import datetime
import decimal
from twisted.internet.testing import StringTransport
from twisted.spread import banana, jelly, pb
from twisted.trial import unittest
from twisted.trial.unittest import TestCase
def test_classSecurity(self):
    """
        Test for class-level security of serialization.
        """
    taster = jelly.SecurityOptions()
    taster.allowInstancesOf(A, B)
    a = A()
    b = B()
    c = C()
    a.b = b
    a.c = c
    a.x = b
    b.c = c
    friendly = jelly.jelly(a, taster)
    x = jelly.unjelly(friendly, taster)
    self.assertIsInstance(x.c, jelly.Unpersistable)
    mean = jelly.jelly(a)
    self.assertRaises(jelly.InsecureJelly, jelly.unjelly, mean, taster)
    self.assertIs(x.x, x.b, 'Identity mismatch')
    friendly = jelly.jelly(A, taster)
    x = jelly.unjelly(friendly, taster)
    self.assertIs(x, A, 'A came back: %s' % x)