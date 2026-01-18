from zope.interface import implementer
from zope.interface.exceptions import BrokenImplementation
from zope.interface.verify import verifyObject
from twisted.application.service import (
from twisted.persisted.sob import IPersistable
from twisted.trial.unittest import TestCase
def stopService(self) -> None:
    """
        See L{twisted.application.service.IService}.
        """