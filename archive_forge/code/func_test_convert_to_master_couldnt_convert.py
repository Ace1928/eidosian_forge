import sys
import unittest
from unittest.mock import MagicMock
from libcloud.test import MockHttp
from libcloud.dns.base import Zone, Record
from libcloud.dns.types import (
from libcloud.utils.py3 import httplib
from libcloud.test.secrets import DNS_PARAMS_ZONOMI
from libcloud.dns.drivers.zonomi import ZonomiDNSDriver
from libcloud.test.file_fixtures import DNSFileFixtures
def test_convert_to_master_couldnt_convert(self):
    zone = self.test_zone
    ZonomiMockHttp.type = 'COULDNT_CONVERT'
    try:
        self.driver.ex_convert_to_master(zone)
    except ZoneDoesNotExistError as e:
        self.assertEqual(e.zone_id, 'zone.com')
    else:
        self.fail('Exception was not thrown.')