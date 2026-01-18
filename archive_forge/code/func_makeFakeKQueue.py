from __future__ import annotations
import errno
from zope.interface import implementer
from twisted.trial.unittest import TestCase
def makeFakeKQueue(testKQueue: object, testKEvent: object) -> _IKQueue:
    """
    Create a fake that implements L{_IKQueue}.

    @param testKQueue: Something that acts like L{select.kqueue}.
    @param testKEvent: Something that acts like L{select.kevent}.
    @return: An implementation of L{_IKQueue} that includes C{testKQueue} and
        C{testKEvent}.
    """

    @implementer(_IKQueue)
    class FakeKQueue:
        kqueue = testKQueue
        kevent = testKEvent
    return FakeKQueue()