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
def test_ex_create_ptr_success(self):
    ip = '127.1.1.1'
    domain = 'www.foo4.bar.com'
    record = self.driver.ex_create_ptr_record(RDNS_NODE, ip, domain)
    self.assertEqual(record.ip, ip)
    self.assertEqual(record.domain, domain)
    self.assertEqual(record.extra['uri'], RDNS_NODE.extra['uri'])
    self.assertEqual(record.extra['service_name'], RDNS_NODE.extra['service_name'])
    self.driver.ex_create_ptr_record(RDNS_LB, ip, domain)