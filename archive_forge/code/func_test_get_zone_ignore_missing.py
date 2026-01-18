from unittest import mock
from os_win import constants
from os_win import exceptions
from os_win.tests.unit import test_base
from os_win.utils.dns import dnsutils
def test_get_zone_ignore_missing(self):
    zone_manager = self._dnsutils._dns_manager.MicrosoftDNS_Zone
    zone_manager.return_value = []
    zone_found = self._dnsutils._get_zone(mock.sentinel.zone_name)
    zone_manager.assert_called_once_with(Name=mock.sentinel.zone_name)
    self.assertIsNone(zone_found)