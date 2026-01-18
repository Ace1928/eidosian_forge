import datetime
import decimal
from twisted.internet.testing import StringTransport
from twisted.spread import banana, jelly, pb
from twisted.trial import unittest
from twisted.trial.unittest import TestCase
def test_newStyleClassesAttributes(self):
    n = TestNode()
    n1 = TestNode(n)
    TestNode(n1)
    TestNode(n)
    jel = jelly.jelly(n)
    m = jelly.unjelly(jel)
    self._check_newstyle(n, m)