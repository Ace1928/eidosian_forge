import sys
import json
from libcloud.test import MockHttp, LibcloudTestCase, unittest
from libcloud.dns.base import Zone
from libcloud.dns.types import (
from libcloud.utils.py3 import httplib
from libcloud.common.types import ProviderError
from libcloud.test.secrets import DNS_PARAMS_AURORADNS
from libcloud.test.file_fixtures import DNSFileFixtures
from libcloud.dns.drivers.auroradns import AuroraDNSDriver, AuroraDNSHealthCheckType
def test_list_records_non_exist(self):
    try:
        self.driver.list_records(Zone(id=1, domain='nonexists.example.com', type='NATIVE', driver=AuroraDNSDriver, ttl=3600))
        self.fail('expected a ZoneDoesNotExistError')
    except ZoneDoesNotExistError:
        pass
    except Exception:
        raise