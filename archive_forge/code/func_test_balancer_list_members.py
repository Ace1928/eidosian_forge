import sys
import unittest
from libcloud.test.secrets import GCE_PARAMS, GCE_KEYWORD_PARAMS
from libcloud.common.google import GoogleBaseAuthConnection
from libcloud.compute.drivers.gce import GCENodeDriver
from libcloud.test.compute.test_gce import GCEMockHttp
from libcloud.test.common.test_google import GoogleTestCase, GoogleAuthMockHttp
from libcloud.loadbalancer.drivers.gce import GCELBDriver
def test_balancer_list_members(self):
    balancer = self.driver.get_balancer('lcforwardingrule')
    members = balancer.list_members()
    self.assertEqual(len(members), 2)
    member_ips = [m.ip for m in members]
    self.assertTrue('23.236.58.15' in member_ips)