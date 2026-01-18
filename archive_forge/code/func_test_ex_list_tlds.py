import sys
import unittest
from libcloud.test import MockHttp
from libcloud.dns.base import Zone, RecordType
from libcloud.utils.py3 import httplib
from libcloud.test.secrets import DNS_PARAMS_GODADDY
from libcloud.test.file_fixtures import DNSFileFixtures
from libcloud.dns.drivers.godaddy import GoDaddyDNSDriver
def test_ex_list_tlds(self):
    tlds = self.driver.ex_list_tlds()
    self.assertEqual(len(tlds), 331)
    self.assertEqual(tlds[0].name, 'academy')
    self.assertEqual(tlds[0].type, 'GENERIC')