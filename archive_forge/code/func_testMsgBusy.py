from six.moves import range
import sys
import mock
from pyu2f import errors
from pyu2f import hidtransport
from pyu2f.tests.lib import util
def testMsgBusy(self):
    fake_hid_dev = util.FakeHidDevice(bytearray([0, 0, 0, 1]), bytearray([1, 144, 0]))
    t = hidtransport.UsbHidTransport(fake_hid_dev)
    fake_hid_dev.SetChannelBusyCount(3)
    with mock.patch.object(hidtransport, 'time') as _:
        self.assertRaisesRegexp(errors.HidError, '^Device Busy', t.SendMsgBytes, [0, 1, 0, 0])
        reply = t.SendMsgBytes([0, 1, 0, 0])
        self.assertEquals(reply, bytearray([1, 144, 0]))