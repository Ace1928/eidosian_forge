import sys
import datetime
from libcloud.test import MockHttp, unittest
from libcloud.utils.py3 import httplib, urlencode
from libcloud.common.types import LibcloudError
from libcloud.loadbalancer.base import Member, Algorithm, LoadBalancer
from libcloud.loadbalancer.types import MemberCondition
from libcloud.test.file_fixtures import OpenStackFixtures, LoadBalancerFileFixtures
from libcloud.loadbalancer.drivers.rackspace import (
def test_get_balancer_extra_created(self):
    balancer = self.driver.get_balancer(balancer_id='8290')
    created_8290 = datetime.datetime(2011, 4, 7, 16, 27, 50)
    self.assertEqual(created_8290, balancer.extra['created'])