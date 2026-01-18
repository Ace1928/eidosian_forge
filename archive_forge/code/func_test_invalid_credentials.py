import sys
import unittest
from libcloud.test import MockHttp
from libcloud.dns.types import RecordType, ZoneDoesNotExistError, RecordDoesNotExistError
from libcloud.utils.py3 import httplib
from libcloud.common.types import LibcloudError, InvalidCredsError
from libcloud.test.secrets import DNS_PARAMS_ZERIGO
from libcloud.dns.drivers.zerigo import ZerigoError, ZerigoDNSDriver
from libcloud.test.file_fixtures import DNSFileFixtures
def test_invalid_credentials(self):
    ZerigoMockHttp.type = 'INVALID_CREDS'
    try:
        list(self.driver.list_zones())
    except InvalidCredsError:
        pass
    else:
        self.fail('Exception was not thrown')