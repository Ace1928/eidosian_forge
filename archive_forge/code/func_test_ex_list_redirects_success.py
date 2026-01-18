import sys
import unittest
from libcloud.test import MockHttp
from libcloud.dns.types import RecordType, ZoneDoesNotExistError, RecordDoesNotExistError
from libcloud.utils.py3 import httplib
from libcloud.test.secrets import DNS_PARAMS_POINTDNS
from libcloud.test.file_fixtures import DNSFileFixtures
from libcloud.dns.drivers.pointdns import PointDNSDriver, PointDNSException
def test_ex_list_redirects_success(self):
    PointDNSMockHttp.type = 'GET'
    zone = self.driver.list_zones()[0]
    PointDNSMockHttp.type = 'LIST'
    redirects = self.driver.ex_list_redirects(zone)
    self.assertEqual(len(redirects), 2)
    redirect1 = redirects[0]
    self.assertEqual(redirect1.id, '36843229')
    self.assertEqual(redirect1.name, 'redirect2.domain1.com.')
    self.assertEqual(redirect1.type, '302')
    self.assertEqual(redirect1.data, 'http://other.com')
    self.assertIsNone(redirect1.iframe)
    self.assertEqual(redirect1.query, False)
    self.assertEqual(zone, redirect1.zone)
    redirect2 = redirects[1]
    self.assertEqual(redirect2.id, '36843497')
    self.assertEqual(redirect2.name, 'redirect1.domain1.com.')
    self.assertEqual(redirect2.type, '302')
    self.assertEqual(redirect2.data, 'http://someother.com')
    self.assertIsNone(redirect2.iframe)
    self.assertEqual(redirect2.query, False)
    self.assertEqual(zone, redirect1.zone)