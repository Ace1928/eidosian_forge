import boto
import time
from tests.compat import unittest
from boto.ec2.elb import ELBConnection
import boto.ec2.elb
def test_create_load_balancer_listeners_with_policies(self):
    more_listeners = [(443, 8001, 'HTTP')]
    self.conn.create_load_balancer_listeners(self.name, more_listeners)
    lb_policy_name = 'lb-policy'
    self.conn.create_lb_cookie_stickiness_policy(1000, self.name, lb_policy_name)
    self.conn.set_lb_policies_of_listener(self.name, self.listeners[0][0], lb_policy_name)
    app_policy_name = 'app-policy'
    self.conn.create_app_cookie_stickiness_policy('appcookie', self.name, app_policy_name)
    self.conn.set_lb_policies_of_listener(self.name, more_listeners[0][0], app_policy_name)
    balancers = self.conn.get_all_load_balancers(load_balancer_names=[self.name])
    self.assertEqual([lb.name for lb in balancers], [self.name])
    self.assertEqual(sorted((l.get_tuple() for l in balancers[0].listeners)), sorted(self.listeners + more_listeners))