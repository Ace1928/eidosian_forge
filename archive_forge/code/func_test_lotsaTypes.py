import datetime
import decimal
from twisted.internet.testing import StringTransport
from twisted.spread import banana, jelly, pb
from twisted.trial import unittest
from twisted.trial.unittest import TestCase
def test_lotsaTypes(self):
    """
        Test for all types currently supported in jelly
        """
    a = A()
    jelly.unjelly(jelly.jelly(a))
    jelly.unjelly(jelly.jelly(a.amethod))
    items = [afunc, [1, 2, 3], not bool(1), bool(1), 'test', 20.3, (1, 2, 3), None, A, unittest, {'a': 1}, A.amethod]
    for i in items:
        self.assertEqual(i, jelly.unjelly(jelly.jelly(i)))