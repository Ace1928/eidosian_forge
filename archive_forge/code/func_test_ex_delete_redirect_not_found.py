import sys
import unittest
from libcloud.test import MockHttp
from libcloud.dns.types import RecordType, ZoneDoesNotExistError, RecordDoesNotExistError
from libcloud.utils.py3 import httplib
from libcloud.test.secrets import DNS_PARAMS_POINTDNS
from libcloud.test.file_fixtures import DNSFileFixtures
from libcloud.dns.drivers.pointdns import PointDNSDriver, PointDNSException
def test_ex_delete_redirect_not_found(self):
    PointDNSMockHttp.type = 'GET'
    zone = self.driver.list_zones()[0]
    redirect = self.driver.ex_get_redirect(zone.id, '36843229')
    PointDNSMockHttp.type = 'DELETE_NOT_FOUND'
    try:
        self.driver.ex_delete_redirect(redirect)
    except PointDNSException as e:
        self.assertEqual(e.http_code, httplib.NOT_FOUND)
        self.assertEqual(e.value, "Couldn't found redirect")
    else:
        self.fail('Exception was not thrown')