import sys
import unittest
from libcloud.test.secrets import GCE_PARAMS, GCE_KEYWORD_PARAMS
from libcloud.common.google import GoogleBaseAuthConnection
from libcloud.compute.drivers.gce import GCENodeDriver
from libcloud.test.compute.test_gce import GCEMockHttp
from libcloud.test.common.test_google import GoogleTestCase, GoogleAuthMockHttp
from libcloud.loadbalancer.drivers.gce import GCELBDriver
def test_node_to_member(self):
    node = self.driver.gce.ex_get_node('libcloud-lb-demo-www-001', 'us-central1-b')
    balancer = self.driver.get_balancer('lcforwardingrule')
    member = self.driver._node_to_member(node, balancer)
    self.assertEqual(member.ip, node.public_ips[0])
    self.assertEqual(member.id, node.name)
    self.assertEqual(member.port, balancer.port)