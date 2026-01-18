import sys
import unittest
from libcloud.test import MockHttp
from libcloud.dns.base import Zone, Record
from libcloud.dns.types import (
from libcloud.utils.py3 import httplib
from libcloud.test.secrets import DNS_PARAMS_DNSPOD
from libcloud.dns.drivers.dnspod import DNSPodDNSDriver
from libcloud.test.file_fixtures import DNSFileFixtures
def test_create_record_already_exists_error(self):
    DNSPodMockHttp.type = 'RECORD_EXISTS'
    try:
        self.driver.create_record(name='@', zone=self.test_zone, type='A', data='92.126.115.73', extra={'ttl': 13, 'record_line': 'default'})
    except RecordAlreadyExistsError as e:
        self.assertEqual(e.value, '@')
    else:
        self.fail('Exception was not thrown')