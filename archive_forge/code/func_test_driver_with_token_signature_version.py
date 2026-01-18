import sys
import unittest
from libcloud.test import MockHttp
from libcloud.utils.py3 import httplib
from libcloud.test.secrets import LB_ALB_PARAMS
from libcloud.loadbalancer.base import Member
from libcloud.loadbalancer.types import State
from libcloud.test.file_fixtures import LoadBalancerFileFixtures
from libcloud.loadbalancer.drivers.alb import ApplicationLBDriver
def test_driver_with_token_signature_version(self):
    token = 'temporary_credentials_token'
    driver = ApplicationLBDriver(*LB_ALB_PARAMS, **{'token': token})
    kwargs = driver._ex_connection_class_kwargs()
    self.assertTrue('signature_version' in kwargs, 'Driver has no attribute signature_version')
    self.assertEqual('4', kwargs['signature_version'], 'Signature version is not 4 with temporary credentials')