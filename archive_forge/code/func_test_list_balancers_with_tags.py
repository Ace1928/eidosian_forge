import sys
import unittest
from libcloud.test import MockHttp
from libcloud.utils.py3 import httplib
from libcloud.test.secrets import LB_ELB_PARAMS
from libcloud.loadbalancer.base import Member, Algorithm
from libcloud.loadbalancer.types import State
from libcloud.test.file_fixtures import LoadBalancerFileFixtures
from libcloud.loadbalancer.drivers.elb import ElasticLBDriver
def test_list_balancers_with_tags(self):
    balancers = self.driver.list_balancers(ex_fetch_tags=True)
    self.assertEqual(len(balancers), 1)
    self.assertEqual(balancers[0].id, 'tests')
    self.assertEqual(balancers[0].name, 'tests')
    self.assertTrue('tags' in balancers[0].extra, 'No tags dict found in balancer.extra')
    self.assertEqual(balancers[0].extra['tags']['project'], 'lima')