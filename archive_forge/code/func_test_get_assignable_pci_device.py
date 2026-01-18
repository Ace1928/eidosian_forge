from unittest import mock
import ddt
import six
from os_win import constants
from os_win import exceptions
from os_win.tests.unit import test_base
from os_win.utils import _wqlutils
from os_win.utils.compute import vmutils10
from os_win.utils import jobutils
def test_get_assignable_pci_device(self):
    pci_dev = mock.MagicMock(DeviceID=self._FAKE_PCI_ID)
    self._vmutils._conn.Msvm_PciExpress.return_value = [pci_dev]
    result = self._vmutils._get_assignable_pci_device(self._FAKE_VENDOR_ID, self._FAKE_PRODUCT_ID)
    self.assertEqual(pci_dev, result)
    self._vmutils._conn.Msvm_PciExpress.assert_called_once_with()