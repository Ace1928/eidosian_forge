import sys
import unittest
from libcloud.test import MockHttp
from libcloud.dns.types import RecordType
from libcloud.utils.py3 import httplib
from libcloud.test.secrets import DNS_PARAMS_LINODE
from libcloud.dns.drivers.linode import LinodeDNSDriver, LinodeDNSDriverV4
from libcloud.test.file_fixtures import DNSFileFixtures
def test_get_zone_not_found(self):
    LinodeMockHttpV4.type = 'ZONE_DOES_NOT_EXIST'
    with self.assertRaises(Exception):
        self.driver.get_zone('123')