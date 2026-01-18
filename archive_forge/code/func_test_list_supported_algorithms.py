import sys
from libcloud.test import MockHttp, unittest
from libcloud.utils.py3 import httplib, urlparse, parse_qsl, assertRaisesRegex
from libcloud.loadbalancer.base import Member, Algorithm, LoadBalancer
from libcloud.loadbalancer.types import Provider
from libcloud.test.file_fixtures import LoadBalancerFileFixtures
from libcloud.loadbalancer.providers import get_driver
from libcloud.loadbalancer.drivers.cloudstack import CloudStackLBDriver
def test_list_supported_algorithms(self):
    algorithms = self.driver.list_supported_algorithms()
    self.assertTrue(Algorithm.ROUND_ROBIN in algorithms)
    self.assertTrue(Algorithm.LEAST_CONNECTIONS in algorithms)