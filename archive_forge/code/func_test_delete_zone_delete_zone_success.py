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
def test_delete_zone_delete_zone_success(self):
    ZonomiMockHttp.type = 'DELETE_ZONE_SUCCESS'
    status = self.driver.delete_zone(zone=self.test_zone)
    self.assertEqual(status, True)