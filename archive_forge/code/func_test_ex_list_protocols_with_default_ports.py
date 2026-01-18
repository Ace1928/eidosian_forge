import sys
import datetime
from libcloud.test import MockHttp, unittest
from libcloud.utils.py3 import httplib, urlencode
from libcloud.common.types import LibcloudError
from libcloud.loadbalancer.base import Member, Algorithm, LoadBalancer
from libcloud.loadbalancer.types import MemberCondition
from libcloud.test.file_fixtures import OpenStackFixtures, LoadBalancerFileFixtures
from libcloud.loadbalancer.drivers.rackspace import (
def test_ex_list_protocols_with_default_ports(self):
    protocols = self.driver.ex_list_protocols_with_default_ports()
    self.assertEqual(len(protocols), 10)
    self.assertTrue(('http', 80) in protocols)