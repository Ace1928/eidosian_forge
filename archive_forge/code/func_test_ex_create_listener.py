import sys
import unittest
from libcloud.test import MockHttp
from libcloud.utils.py3 import httplib
from libcloud.test.secrets import LB_ALB_PARAMS
from libcloud.loadbalancer.base import Member
from libcloud.loadbalancer.types import State
from libcloud.test.file_fixtures import LoadBalancerFileFixtures
from libcloud.loadbalancer.drivers.alb import ApplicationLBDriver
def test_ex_create_listener(self):
    balancer = self.driver.get_balancer(self.balancer_id)
    target_group = self.driver.ex_get_target_group(self.target_group_id)
    listener = self.driver.ex_create_listener(balancer=balancer, port=443, proto='HTTPS', target_group=target_group, action='forward', ssl_cert_arn=self.ssl_cert_id, ssl_policy='ELBSecurityPolicy-2016-08')
    self.assertTrue(hasattr(listener, 'id'), 'Listener is missing "id" field')
    self.assertTrue(hasattr(listener, 'rules'), 'Listener is missing "rules" field')
    self.assertTrue(hasattr(listener, 'balancer'), 'Listener is missing "balancer" field')
    self.assertEqual(listener.balancer.id, balancer.id)
    self.assertEqual(listener.rules[0].target_group.id, target_group.id)
    self.assertEqual(listener.port, 443)
    self.assertEqual(listener.protocol, 'HTTPS')
    self.assertEqual(listener.action, 'forward')
    self.assertEqual(listener.ssl_certificate, self.ssl_cert_id)
    self.assertEqual(listener.ssl_policy, 'ELBSecurityPolicy-2016-08')