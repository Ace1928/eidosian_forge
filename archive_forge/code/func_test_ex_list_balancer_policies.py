import sys
import unittest
from libcloud.test import MockHttp
from libcloud.utils.py3 import httplib
from libcloud.test.secrets import LB_ELB_PARAMS
from libcloud.loadbalancer.base import Member, Algorithm
from libcloud.loadbalancer.types import State
from libcloud.test.file_fixtures import LoadBalancerFileFixtures
from libcloud.loadbalancer.drivers.elb import ElasticLBDriver
def test_ex_list_balancer_policies(self):
    balancer = self.driver.get_balancer(balancer_id='tests')
    policies = self.driver.ex_list_balancer_policies(balancer)
    self.assertTrue('MyDurationStickyPolicy' in policies)