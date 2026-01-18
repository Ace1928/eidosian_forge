import sys
import unittest
from libcloud.test import MockHttp
from libcloud.dns.types import RecordType, ZoneDoesNotExistError, RecordDoesNotExistError
from libcloud.utils.py3 import httplib
from libcloud.test.secrets import DNS_PARAMS_POINTDNS
from libcloud.test.file_fixtures import DNSFileFixtures
from libcloud.dns.drivers.pointdns import PointDNSDriver, PointDNSException
def test_ex_create_redirect(self):
    PointDNSMockHttp.type = 'GET'
    zone = self.driver.list_zones()[0]
    PointDNSMockHttp.type = 'CREATE'
    redirect = self.driver.ex_create_redirect('http://other.com', 'redirect2', '302', zone, iframe='An Iframe', query=True)
    self.assertEqual(redirect.id, '36843229')
    self.assertEqual(redirect.name, 'redirect2.domain1.com.')
    self.assertEqual(redirect.type, '302')
    self.assertEqual(redirect.data, 'http://other.com')
    self.assertEqual(redirect.iframe, 'An Iframe')
    self.assertEqual(redirect.query, True)
    self.assertEqual(zone.id, redirect.zone.id)