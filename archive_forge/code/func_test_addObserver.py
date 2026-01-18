from typing import Dict, List, Tuple, cast
from zope.interface import implementer
from zope.interface.exceptions import BrokenMethodImplementation
from zope.interface.verify import verifyObject
from twisted.trial import unittest
from .._interfaces import ILogObserver, LogEvent
from .._logger import Logger
from .._observer import LogPublisher
def test_addObserver(self) -> None:
    """
        L{LogPublisher.addObserver} adds an observer.
        """
    o1 = cast(ILogObserver, lambda e: None)
    o2 = cast(ILogObserver, lambda e: None)
    o3 = cast(ILogObserver, lambda e: None)
    publisher = LogPublisher(o1, o2)
    publisher.addObserver(o3)
    self.assertEqual({o1, o2, o3}, set(publisher._observers))