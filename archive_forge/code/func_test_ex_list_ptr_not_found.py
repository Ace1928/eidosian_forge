import sys
import unittest
from libcloud.test import MockHttp
from libcloud.dns.types import RecordType, ZoneDoesNotExistError, RecordDoesNotExistError
from libcloud.utils.py3 import httplib
from libcloud.common.types import LibcloudError
from libcloud.compute.base import Node
from libcloud.test.secrets import DNS_PARAMS_RACKSPACE
from libcloud.loadbalancer.base import LoadBalancer
from libcloud.test.file_fixtures import DNSFileFixtures
from libcloud.dns.drivers.rackspace import RackspaceDNSDriver, RackspacePTRRecord
def test_ex_list_ptr_not_found(self):
    RackspaceMockHttp.type = 'RECORD_DOES_NOT_EXIST'
    try:
        records = self.driver.ex_iterate_ptr_records(RDNS_NODE)
    except Exception as exc:
        self.fail('PTR Records list 404 threw %s' % exc)
    try:
        next(records)
        self.fail('PTR Records list 404 did not produce an empty list')
    except StopIteration:
        self.assertTrue(True, 'Got empty list on 404')