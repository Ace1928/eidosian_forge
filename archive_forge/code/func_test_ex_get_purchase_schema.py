import sys
import unittest
from libcloud.test import MockHttp
from libcloud.dns.base import Zone, RecordType
from libcloud.utils.py3 import httplib
from libcloud.test.secrets import DNS_PARAMS_GODADDY
from libcloud.test.file_fixtures import DNSFileFixtures
from libcloud.dns.drivers.godaddy import GoDaddyDNSDriver
def test_ex_get_purchase_schema(self):
    schema = self.driver.ex_get_purchase_schema('com')
    self.assertEqual(schema['id'], 'https://api.godaddy.com/DomainPurchase#')