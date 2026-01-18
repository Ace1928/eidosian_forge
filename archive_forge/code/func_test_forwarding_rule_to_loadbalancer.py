import sys
import unittest
from libcloud.test.secrets import GCE_PARAMS, GCE_KEYWORD_PARAMS
from libcloud.common.google import GoogleBaseAuthConnection
from libcloud.compute.drivers.gce import GCENodeDriver
from libcloud.test.compute.test_gce import GCEMockHttp
from libcloud.test.common.test_google import GoogleTestCase, GoogleAuthMockHttp
from libcloud.loadbalancer.drivers.gce import GCELBDriver
def test_forwarding_rule_to_loadbalancer(self):
    fwr = self.driver.gce.ex_get_forwarding_rule('lcforwardingrule')
    balancer = self.driver._forwarding_rule_to_loadbalancer(fwr)
    self.assertEqual(fwr.name, balancer.name)
    self.assertEqual(fwr.address, balancer.ip)
    self.assertEqual(fwr.extra['portRange'], balancer.port)