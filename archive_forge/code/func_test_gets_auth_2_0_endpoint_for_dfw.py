import sys
import datetime
from libcloud.test import MockHttp, unittest
from libcloud.utils.py3 import httplib, urlencode
from libcloud.common.types import LibcloudError
from libcloud.loadbalancer.base import Member, Algorithm, LoadBalancer
from libcloud.loadbalancer.types import MemberCondition
from libcloud.test.file_fixtures import OpenStackFixtures, LoadBalancerFileFixtures
from libcloud.loadbalancer.drivers.rackspace import (
def test_gets_auth_2_0_endpoint_for_dfw(self):
    driver = RackspaceLBDriver('user', 'key', ex_force_auth_version='2.0_password', ex_force_region='dfw')
    driver.connection._populate_hosts_and_request_paths()
    self.assertEqual('https://dfw.loadbalancers.api.rackspacecloud.com/v1.0/11111', driver.connection.get_endpoint())