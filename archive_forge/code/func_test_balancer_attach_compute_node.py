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
def test_balancer_attach_compute_node(self):
    SLBMockHttp.type = 'attach_compute_node'
    self.balancer = self.driver.get_balancer(balancer_id='tests')
    self.node = Node(id='node1', name='node-name', state=NodeState.RUNNING, public_ips=['1.2.3.4'], private_ips=['4.3.2.1'], driver=self.driver)
    member = self.driver.balancer_attach_compute_node(self.balancer, self.node)
    self.assertEqual(self.node.id, member.id)
    self.assertEqual(self.node.public_ips[0], member.ip)
    self.assertEqual(self.balancer.port, member.port)