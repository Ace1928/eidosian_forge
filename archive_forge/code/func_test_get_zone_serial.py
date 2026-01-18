from unittest import mock
from os_win import constants
from os_win import exceptions
from os_win.tests.unit import test_base
from os_win.utils.dns import dnsutils
@mock.patch.object(dnsutils.DNSUtils, 'zone_exists')
def test_get_zone_serial(self, mock_zone_exists):
    mock_zone_exists.return_value = True
    fake_serial_number = 1
    msdns_soatype = self._dnsutils._dns_manager.MicrosoftDNS_SOAType
    msdns_soatype.return_value = [mock.Mock(SerialNumber=fake_serial_number)]
    serial_number = self._dnsutils.get_zone_serial(mock.sentinel.zone_name)
    expected_serial_number = fake_serial_number
    self.assertEqual(expected_serial_number, serial_number)
    msdns_soatype.assert_called_once_with(ContainerName=mock.sentinel.zone_name)
    mock_zone_exists.assert_called_once_with(mock.sentinel.zone_name)