import unittest
from zope.interface.tests import OptimizationTestMixin
def remove_extendor(self, provided):
    self._extendors = tuple([x for x in self._extendors if x != provided])