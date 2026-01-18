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
def test_balancer_list_listeners(self):
    balancer = self.driver.get_balancer(balancer_id='tests')
    listeners = self.driver.ex_list_listeners(balancer)
    self.assertEqual(1, len(listeners))
    listener = listeners[0]
    self.assertEqual('80', listener.port)