import sys
import unittest
from libcloud.test import MockHttp
from libcloud.utils.py3 import httplib
from libcloud.test.secrets import LB_ELB_PARAMS
from libcloud.loadbalancer.base import Member, Algorithm
from libcloud.loadbalancer.types import State
from libcloud.test.file_fixtures import LoadBalancerFileFixtures
from libcloud.loadbalancer.drivers.elb import ElasticLBDriver
def test_ex_set_balancer_policies_backend_server(self):
    self.assertTrue(self.driver.ex_set_balancer_policies_backend_server(name='tests', instance_port=80, policies=['MyDurationProxyPolicy']))