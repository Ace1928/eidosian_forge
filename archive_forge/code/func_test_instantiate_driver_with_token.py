import sys
import unittest
from libcloud.test import MockHttp
from libcloud.utils.py3 import httplib
from libcloud.test.secrets import LB_ALB_PARAMS
from libcloud.loadbalancer.base import Member
from libcloud.loadbalancer.types import State
from libcloud.test.file_fixtures import LoadBalancerFileFixtures
from libcloud.loadbalancer.drivers.alb import ApplicationLBDriver
def test_instantiate_driver_with_token(self):
    token = 'temporary_credentials_token'
    driver = ApplicationLBDriver(*LB_ALB_PARAMS, **{'token': token})
    self.assertTrue(hasattr(driver, 'token'), 'Driver has no attribute token')
    self.assertEqual(token, driver.token, 'Driver token does not match with provided token')