import sys
import unittest
from libcloud.test import MockHttp
from libcloud.dns.types import RecordType, ZoneDoesNotExistError, RecordDoesNotExistError
from libcloud.utils.py3 import httplib
from libcloud.test.secrets import DNS_PARAMS_POINTDNS
from libcloud.test.file_fixtures import DNSFileFixtures
from libcloud.dns.drivers.pointdns import PointDNSDriver, PointDNSException
def test_ex_get_redirect(self):
    PointDNSMockHttp.type = 'GET'
    zone = self.driver.list_zones()[0]
    redirect = self.driver.ex_get_redirect(zone.id, '36843229')
    self.assertEqual(redirect.id, '36843229')
    self.assertEqual(redirect.name, 'redirect2.domain1.com.')
    self.assertEqual(redirect.type, '302')
    self.assertEqual(redirect.data, 'http://other.com')
    self.assertIsNone(redirect.iframe)
    self.assertEqual(redirect.query, False)
    self.assertEqual(zone.id, redirect.zone.id)