import ctypes
import sys
import mock
from pyu2f import errors
from pyu2f.hid import macos
@mock.patch.object(macos.threading, 'Thread')
@mock.patch.multiple(macos, iokit=mock.DEFAULT, cf=mock.DEFAULT, GetDeviceIntProperty=mock.DEFAULT)
def testCallWriteSuccess(self, thread, iokit, cf, GetDeviceIntProperty):
    init_mock_iokit(iokit)
    init_mock_cf(cf)
    init_mock_get_int_property(GetDeviceIntProperty)
    device = macos.MacOsHidDevice('fakepath')
    data = bytearray(range(64))
    device.Write(data)
    set_report_call_args = iokit.IOHIDDeviceSetReport.call_args
    self.assertIsNotNone(set_report_call_args)
    set_report_call_pos_args = iokit.IOHIDDeviceSetReport.call_args[0]
    self.assertEquals(len(set_report_call_pos_args), 5)
    self.assertEquals(set_report_call_pos_args[0], 'handle')
    self.assertEquals(set_report_call_pos_args[1], 1)
    self.assertEquals(set_report_call_pos_args[2], 0)
    self.assertEquals(set_report_call_pos_args[4], 64)
    report_buffer = set_report_call_pos_args[3]
    self.assertEqual(len(report_buffer), 64)
    self.assertEqual(bytearray(report_buffer), data, 'Data sent to IOHIDDeviceSetReport should match data sent to the device')