import sys
import json
from libcloud.test import MockHttp, unittest
from libcloud.dns.types import RecordType, ZoneDoesNotExistError, RecordDoesNotExistError
from libcloud.utils.py3 import httplib, urlparse
from libcloud.common.types import LibcloudError
from libcloud.test.secrets import DNS_PARAMS_CLOUDFLARE
from libcloud.test.file_fixtures import DNSFileFixtures
from libcloud.dns.drivers.cloudflare import (
def test_create_record_error_with_error_chain(self):
    zone = self.driver.list_zones()[0]
    CloudFlareMockHttp.type = 'error_chain_error'
    expected_msg = '.*1004: DNS Validation Error \\(error chain: 9011: Length of content is invalid\\)'
    self.assertRaisesRegex(LibcloudError, expected_msg, self.driver.create_record, name='test5', zone=zone, type=RecordType.CAA, data='caa.foo.com')