from unittest import mock
from os_win import constants
from os_win import exceptions
from os_win.tests.unit import test_base
from os_win.utils.dns import dnsutils
@mock.patch.object(dnsutils.DNSUtils, 'zone_exists')
def test_get_zone_serial_zone_soatype_not_found(self, mock_zone_exists):
    mock_zone_exists.return_value = True
    self._dnsutils._dns_manager.MicrosoftDNS_SOAType.return_value = []
    serial_number = self._dnsutils.get_zone_serial(mock.sentinel.zone_name)
    self.assertIsNone(serial_number)
    mock_zone_exists.assert_called_once_with(mock.sentinel.zone_name)