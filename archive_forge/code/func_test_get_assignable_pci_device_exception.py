from unittest import mock
import ddt
import six
from os_win import constants
from os_win import exceptions
from os_win.tests.unit import test_base
from os_win.utils import _wqlutils
from os_win.utils.compute import vmutils10
from os_win.utils import jobutils
@ddt.data(True, False)
def test_get_assignable_pci_device_exception(self, matched):
    product_id = self._FAKE_PRODUCT_ID if matched else '0000'
    pci_dev = mock.MagicMock(DeviceID=self._FAKE_PCI_ID)
    pci_devs = [pci_dev] * 2 if matched else [pci_dev]
    self._vmutils._conn.Msvm_PciExpress.return_value = pci_devs
    self.assertRaises(exceptions.PciDeviceNotFound, self._vmutils._get_assignable_pci_device, self._FAKE_VENDOR_ID, product_id)
    self._vmutils._conn.Msvm_PciExpress.assert_called_once_with()