import sys
import unittest
from libcloud.test.secrets import GCE_PARAMS, GCE_KEYWORD_PARAMS
from libcloud.common.google import GoogleBaseAuthConnection
from libcloud.compute.drivers.gce import GCENodeDriver
from libcloud.test.compute.test_gce import GCEMockHttp
from libcloud.test.common.test_google import GoogleTestCase, GoogleAuthMockHttp
from libcloud.loadbalancer.drivers.gce import GCELBDriver
def test_list_balancers(self):
    balancers = self.driver.list_balancers()
    balancers_all = self.driver.list_balancers(ex_region='all')
    balancer_name = 'lcforwardingrule'
    self.assertEqual(len(balancers), 2)
    self.assertEqual(len(balancers_all), 2)
    self.assertEqual(balancers[0].name, balancer_name)