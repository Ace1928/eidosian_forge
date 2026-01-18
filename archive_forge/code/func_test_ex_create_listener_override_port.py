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
def test_ex_create_listener_override_port(self):
    SLBMockHttp.type = 'create_listener_override_port'
    self.balancer = self.driver.get_balancer(balancer_id='tests')
    self.backend_port = 80
    self.protocol = 'http'
    self.algorithm = Algorithm.WEIGHTED_ROUND_ROBIN
    self.bandwidth = 1
    self.extra = {'StickySession': 'off', 'HealthCheck': 'off', 'ListenerPort': 8080}
    self.assertTrue(self.driver.ex_create_listener(self.balancer, self.backend_port, self.protocol, self.algorithm, self.bandwidth, **self.extra))