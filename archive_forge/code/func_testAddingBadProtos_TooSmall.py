from __future__ import annotations
from zope import interface
from twisted.pair import ip, raw
from twisted.python import components
from twisted.trial import unittest
def testAddingBadProtos_TooSmall(self) -> None:
    """Adding a protocol with a negative number raises an exception."""
    e = ip.IPProtocol()
    try:
        e.addProto(-1, MyProtocol([]))
    except TypeError as e:
        if e.args == ('Added protocol must be positive or zero',):
            pass
        else:
            raise
    else:
        raise AssertionError('addProto must raise an exception for bad protocols')