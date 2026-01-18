from __future__ import annotations
from zope import interface
from twisted.pair import ip, raw
from twisted.python import components
from twisted.trial import unittest
def testDemuxing(self) -> None:
    proto = ip.IPProtocol()
    p1 = MyProtocol([(b'foobar', {'partial': 0, 'dest': '1.2.3.4', 'source': '5.6.7.8', 'protocol': 15, 'version': 4, 'ihl': 20, 'tos': 7, 'tot_len': 20 + 6, 'fragment_id': 57005, 'fragment_offset': 7919, 'dont_fragment': 0, 'more_fragments': 1, 'ttl': 192}), (b'quux', {'partial': 1, 'dest': '5.4.3.2', 'source': '6.7.8.9', 'protocol': 15, 'version': 4, 'ihl': 20, 'tos': 7, 'tot_len': 20 + 6, 'fragment_id': 57005, 'fragment_offset': 7919, 'dont_fragment': 0, 'more_fragments': 1, 'ttl': 192})])
    proto.addProto(15, p1)
    p2 = MyProtocol([(b'quux', {'partial': 1, 'dest': '5.4.3.2', 'source': '6.7.8.9', 'protocol': 10, 'version': 4, 'ihl': 20, 'tos': 7, 'tot_len': 20 + 6, 'fragment_id': 57005, 'fragment_offset': 7919, 'dont_fragment': 0, 'more_fragments': 1, 'ttl': 192}), (b'foobar', {'partial': 0, 'dest': '1.2.3.4', 'source': '5.6.7.8', 'protocol': 10, 'version': 4, 'ihl': 20, 'tos': 7, 'tot_len': 20 + 6, 'fragment_id': 57005, 'fragment_offset': 7919, 'dont_fragment': 0, 'more_fragments': 1, 'ttl': 192})])
    proto.addProto(10, p2)
    proto.datagramReceived(b'T' + b'\x07' + b'\x00\x1a' + b'\xde\xad' + b'\xbe\xef' + b'\xc0' + b'\n' + b'FE' + b'\x06\x07\x08\t' + b'\x05\x04\x03\x02' + b'quux', partial=1, dest='dummy', source='dummy', protocol='dummy')
    proto.datagramReceived(b'T' + b'\x07' + b'\x00\x1a' + b'\xde\xad' + b'\xbe\xef' + b'\xc0' + b'\x0f' + b'FE' + b'\x05\x06\x07\x08' + b'\x01\x02\x03\x04' + b'foobar', partial=0, dest='dummy', source='dummy', protocol='dummy')
    proto.datagramReceived(b'T' + b'\x07' + b'\x00\x1a' + b'\xde\xad' + b'\xbe\xef' + b'\xc0' + b'\x0f' + b'FE' + b'\x06\x07\x08\t' + b'\x05\x04\x03\x02' + b'quux', partial=1, dest='dummy', source='dummy', protocol='dummy')
    proto.datagramReceived(b'T' + b'\x07' + b'\x00\x1a' + b'\xde\xad' + b'\xbe\xef' + b'\xc0' + b'\n' + b'FE' + b'\x05\x06\x07\x08' + b'\x01\x02\x03\x04' + b'foobar', partial=0, dest='dummy', source='dummy', protocol='dummy')
    assert not p1.expecting, 'Should not expect any more packets, but still want %r' % p1.expecting
    assert not p2.expecting, 'Should not expect any more packets, but still want %r' % p2.expecting