from six.moves import range
import sys
import mock
from pyu2f import errors
from pyu2f import hidtransport
from pyu2f.tests.lib import util
def testContPacketShape(self):
    packet = hidtransport.UsbHidTransport.ContPacket(64, bytearray(b'\x00\x00\x00\x01'), 5, bytearray(b'\x01\x02'))
    self.assertEquals(packet.ToWireFormat(), RPad([0, 0, 0, 1, 5, 1, 2], 64))
    copy = hidtransport.UsbHidTransport.ContPacket.FromWireFormat(64, packet.ToWireFormat())
    self.assertEquals(copy.cid, bytearray(b'\x00\x00\x00\x01'))
    self.assertEquals(copy.seq, 5)
    self.assertEquals(copy.payload, RPad(bytearray(b'\x01\x02'), 59))