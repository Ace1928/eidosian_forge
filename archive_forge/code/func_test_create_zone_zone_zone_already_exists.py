import sys
import unittest
from libcloud.test import MockHttp
from libcloud.dns.base import Zone, Record
from libcloud.dns.types import (
from libcloud.utils.py3 import httplib
from libcloud.test.secrets import DNS_PARAMS_LIQUIDWEB
from libcloud.test.file_fixtures import DNSFileFixtures
from libcloud.dns.drivers.liquidweb import LiquidWebDNSDriver
def test_create_zone_zone_zone_already_exists(self):
    LiquidWebMockHttp.type = 'CREATE_ZONE_ZONE_ALREADY_EXISTS'
    try:
        self.driver.create_zone(domain='test.com')
    except ZoneAlreadyExistsError as e:
        self.assertEqual(e.zone_id, 'test.com')
    else:
        self.fail('Exception was not thrown')