import sys
import datetime
from libcloud.test import MockHttp, unittest
from libcloud.utils.py3 import httplib, urlencode
from libcloud.common.types import LibcloudError
from libcloud.loadbalancer.base import Member, Algorithm, LoadBalancer
from libcloud.loadbalancer.types import MemberCondition
from libcloud.test.file_fixtures import OpenStackFixtures, LoadBalancerFileFixtures
from libcloud.loadbalancer.drivers.rackspace import (
def test_ex_list_current_usage(self):
    balancer = self.driver.get_balancer(balancer_id='8290')
    usage = self.driver.ex_list_current_usage(balancer=balancer)
    self.assertEqual(usage['loadBalancerUsageRecords'][0]['incomingTransferSsl'], 6182163)