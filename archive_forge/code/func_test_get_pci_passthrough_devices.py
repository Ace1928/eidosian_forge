from unittest import mock
from os_win import constants
from os_win import exceptions
from os_win.tests.unit import test_base
from os_win.utils import hostutils
def test_get_pci_passthrough_devices(self):
    self.assertEqual([], self._hostutils.get_pci_passthrough_devices())