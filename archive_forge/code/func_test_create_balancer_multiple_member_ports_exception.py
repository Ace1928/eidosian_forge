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
def test_create_balancer_multiple_member_ports_exception(self):
    members = [Member('m1', '1.2.3.4', 80), Member('m2', '1.2.3.5', 81)]
    self.assertRaises(AttributeError, self.driver.create_balancer, None, 80, 'http', Algorithm.WEIGHTED_ROUND_ROBIN, members)