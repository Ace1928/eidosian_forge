import boto
import time
from tests.compat import unittest
from boto.ec2.elb import ELBConnection
import boto.ec2.elb
def test_create_load_balancer_listeners(self):
    more_listeners = [(443, 8001, 'HTTP')]
    self.conn.create_load_balancer_listeners(self.name, more_listeners)
    balancers = self.conn.get_all_load_balancers()
    self.assertEqual([lb.name for lb in balancers], [self.name])
    self.assertEqual(sorted((l.get_tuple() for l in balancers[0].listeners)), sorted(self.listeners + more_listeners))