from zope.interface import implementer
from twisted.trial import unittest
from .._interfaces import ILogObserver, LogEvent
from .._observer import LogPublisher
from .._util import formatTrace
@implementer(ILogObserver)
def o3(e: LogEvent) -> None:
    pass