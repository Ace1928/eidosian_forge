import unittest
from zope.interface import Interface
from zope.interface.adapter import VerifyingAdapterRegistry
from zope.interface.registry import Components
def registerUtility(self, *args):
    self.reg_count += 1
    if self.reg_count == 2:
        self._utility_registrations = dict(self._utility_registrations)
    super().registerUtility(*args)