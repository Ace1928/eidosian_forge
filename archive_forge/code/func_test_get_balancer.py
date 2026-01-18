import sys
import unittest
from libcloud.test.secrets import GCE_PARAMS, GCE_KEYWORD_PARAMS
from libcloud.common.google import GoogleBaseAuthConnection
from libcloud.compute.drivers.gce import GCENodeDriver
from libcloud.test.compute.test_gce import GCEMockHttp
from libcloud.test.common.test_google import GoogleTestCase, GoogleAuthMockHttp
from libcloud.loadbalancer.drivers.gce import GCELBDriver
def test_get_balancer(self):
    balancer_name = 'lcforwardingrule'
    tp_name = 'lctargetpool'
    balancer_ip = '173.255.119.224'
    balancer = self.driver.get_balancer(balancer_name)
    self.assertEqual(balancer.name, balancer_name)
    self.assertEqual(balancer.extra['forwarding_rule'].name, balancer_name)
    self.assertEqual(balancer.ip, balancer_ip)
    self.assertEqual(balancer.extra['targetpool'].name, tp_name)