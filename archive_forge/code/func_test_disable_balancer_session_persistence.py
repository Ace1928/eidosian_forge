import sys
import datetime
from libcloud.test import MockHttp, unittest
from libcloud.utils.py3 import httplib, urlencode
from libcloud.common.types import LibcloudError
from libcloud.loadbalancer.base import Member, Algorithm, LoadBalancer
from libcloud.loadbalancer.types import MemberCondition
from libcloud.test.file_fixtures import OpenStackFixtures, LoadBalancerFileFixtures
from libcloud.loadbalancer.drivers.rackspace import (
def test_disable_balancer_session_persistence(self):
    balancer = self.driver.get_balancer(balancer_id='8290')
    balancer = self.driver.ex_disable_balancer_session_persistence(balancer)
    self.assertTrue('sessionPersistenceType' not in balancer.extra)