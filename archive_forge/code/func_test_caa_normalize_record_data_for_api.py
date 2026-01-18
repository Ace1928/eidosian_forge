import sys
import json
from libcloud.test import MockHttp, unittest
from libcloud.dns.types import RecordType, ZoneDoesNotExistError, RecordDoesNotExistError
from libcloud.utils.py3 import httplib, urlparse
from libcloud.common.types import LibcloudError
from libcloud.test.secrets import DNS_PARAMS_CLOUDFLARE
from libcloud.test.file_fixtures import DNSFileFixtures
from libcloud.dns.drivers.cloudflare import (
def test_caa_normalize_record_data_for_api(self):
    content, data = self.driver._normalize_record_data_for_api(RecordType.CAA, '0 issue foo.bar')
    self.assertEqual(content, '0\tissue\tfoo.bar')
    self.assertEqual(data, {})