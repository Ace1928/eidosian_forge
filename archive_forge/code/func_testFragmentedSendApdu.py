from six.moves import range
import sys
import mock
from pyu2f import errors
from pyu2f import hidtransport
from pyu2f.tests.lib import util
def testFragmentedSendApdu(self):
    body = bytearray((x % 256 for x in range(0, 1000)))
    fake_hid_dev = util.FakeHidDevice(bytearray([0, 0, 0, 1]), [144, 0])
    t = hidtransport.UsbHidTransport(fake_hid_dev)
    reply = t.SendMsgBytes(body)
    self.assertEquals(reply, bytearray([144, 0]))
    self.assertEquals(len(fake_hid_dev.received_packets), 18)