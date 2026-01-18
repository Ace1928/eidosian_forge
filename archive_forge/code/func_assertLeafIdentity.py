import unittest
from zope.interface.tests import OptimizationTestMixin
def assertLeafIdentity(self, leaf1, leaf2):
    self.assertIs(leaf1, leaf2)