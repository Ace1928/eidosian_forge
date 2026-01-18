import sys
import unittest
from libcloud.test import MockHttp
from libcloud.utils.py3 import httplib
from libcloud.compute.base import Node
from libcloud.test.secrets import LB_SLB_PARAMS
from libcloud.compute.types import NodeState
from libcloud.loadbalancer.base import Member, Algorithm
from libcloud.loadbalancer.types import State
from libcloud.test.file_fixtures import LoadBalancerFileFixtures
from libcloud.loadbalancer.drivers.slb import (
def test_ex_start_listener(self):
    SLBMockHttp.type = 'start_listener'
    balancer = self.driver.get_balancer(balancer_id='tests')
    self.port = 80
    self.assertTrue(self.driver.ex_start_listener(balancer, self.port))