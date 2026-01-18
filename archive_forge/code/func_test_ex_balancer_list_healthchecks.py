import sys
import unittest
from libcloud.test.secrets import GCE_PARAMS, GCE_KEYWORD_PARAMS
from libcloud.common.google import GoogleBaseAuthConnection
from libcloud.compute.drivers.gce import GCENodeDriver
from libcloud.test.compute.test_gce import GCEMockHttp
from libcloud.test.common.test_google import GoogleTestCase, GoogleAuthMockHttp
from libcloud.loadbalancer.drivers.gce import GCELBDriver
def test_ex_balancer_list_healthchecks(self):
    balancer = self.driver.get_balancer('lcforwardingrule')
    healthchecks = self.driver.ex_balancer_list_healthchecks(balancer)
    self.assertEqual(healthchecks[0].name, 'libcloud-lb-demo-healthcheck')