from __future__ import annotations
from zope import interface
from twisted.pair import ip, raw
from twisted.python import components
from twisted.trial import unittest
def testWrongProtoNotSeen(self) -> None:
    proto = ip.IPProtocol()
    p1 = MyProtocol([])
    proto.addProto(1, p1)
    proto.datagramReceived(b'T' + b'\x07' + b'\x00\x1a' + b'\xde\xad' + b'\xbe\xef' + b'\xc0' + b'\x0f' + b'FE' + b'\x05\x06\x07\x08' + b'\x01\x02\x03\x04' + b'foobar', partial=0, dest='dummy', source='dummy', protocol='dummy')