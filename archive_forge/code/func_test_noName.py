from zope.interface import implementer
from zope.interface.exceptions import BrokenImplementation
from zope.interface.verify import verifyObject
from twisted.application.service import (
from twisted.persisted.sob import IPersistable
from twisted.trial.unittest import TestCase
def test_noName(self) -> None:
    """
        AlmostService with no name does not implement IService.
        """
    self.almostService.makeInvalidByDeletingName()
    with self.assertRaises(BrokenImplementation):
        verifyObject(IService, self.almostService)