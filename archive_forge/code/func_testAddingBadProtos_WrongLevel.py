from __future__ import annotations
from zope import interface
from twisted.pair import ip, raw
from twisted.python import components
from twisted.trial import unittest
def testAddingBadProtos_WrongLevel(self) -> None:
    """Adding a wrong level protocol raises an exception."""
    e = ip.IPProtocol()
    try:
        e.addProto(42, 'silliness')
    except components.CannotAdapt:
        pass
    else:
        raise AssertionError('addProto must raise an exception for bad protocols')