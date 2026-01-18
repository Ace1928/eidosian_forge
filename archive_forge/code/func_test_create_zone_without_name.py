import sys
import json
import unittest
from libcloud.dns.base import Zone, Record
from libcloud.dns.types import (
from libcloud.utils.py3 import httplib
from libcloud.test.secrets import DNS_GANDI_LIVE
from libcloud.common.gandi_live import JsonParseError, GandiLiveBaseError, InvalidRequestError
from libcloud.test.file_fixtures import DNSFileFixtures
from libcloud.dns.drivers.gandi_live import GandiLiveDNSDriver
from libcloud.test.common.test_gandi_live import BaseGandiLiveMockHttp
def test_create_zone_without_name(self):
    zone = self.driver.create_zone('example.org')
    self.assertEqual(zone.id, 'example.org')
    self.assertEqual(zone.domain, 'example.org')
    self.assertEqual(zone.extra['zone_uuid'], '54321')