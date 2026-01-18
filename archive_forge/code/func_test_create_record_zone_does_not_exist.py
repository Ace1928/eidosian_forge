import sys
import unittest
from unittest.mock import MagicMock
from libcloud.test import MockHttp, LibcloudTestCase
from libcloud.dns.base import Zone, Record
from libcloud.dns.types import (
from libcloud.utils.py3 import httplib
from libcloud.test.secrets import DNS_PARAMS_DURABLEDNS
from libcloud.test.file_fixtures import DNSFileFixtures
from libcloud.dns.drivers.durabledns import (
def test_create_record_zone_does_not_exist(self):
    DurableDNSMockHttp.type = 'ZONE_DOES_NOT_EXIST'
    z_extra = {'ns': 'ns1.durabledns.com.', 'mbox': 'mail.myzone.com', 'refresh': '13000', 'retry': 7200, 'expire': 1300, 'minimum': 13, 'xfer': '127.0.0.1', 'update_acl': '127.0.0.1'}
    zone = Zone(id='deletedzone.com.', domain='deletedzone.com.', type='master', ttl=1300, driver=self.driver, extra=z_extra)
    try:
        self.driver.create_record(name='record1', zone=zone, type=RecordType.A, data='1.2.3.4')
    except ZoneDoesNotExistError:
        pass
    else:
        self.fail('Exception was not thrown')