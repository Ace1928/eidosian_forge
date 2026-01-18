import sys
import unittest
from libcloud.test import MockHttp
from libcloud.dns.base import Zone, Record
from libcloud.dns.types import (
from libcloud.utils.py3 import httplib
from libcloud.test.secrets import DNS_PARAMS_LIQUIDWEB
from libcloud.test.file_fixtures import DNSFileFixtures
from libcloud.dns.drivers.liquidweb import LiquidWebDNSDriver
def test_get_zone_zone_does_not_exist(self):
    LiquidWebMockHttp.type = 'ZONE_DOES_NOT_EXIST'
    try:
        self.driver.get_zone(zone_id='13')
    except ZoneDoesNotExistError as e:
        self.assertEqual(e.zone_id, '13')
    else:
        self.fail('Exception was not thrown')