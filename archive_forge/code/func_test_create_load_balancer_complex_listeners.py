import boto
import time
from tests.compat import unittest
from boto.ec2.elb import ELBConnection
import boto.ec2.elb
def test_create_load_balancer_complex_listeners(self):
    complex_listeners = [(8080, 80, 'HTTP', 'HTTP'), (2525, 25, 'TCP', 'TCP')]
    self.conn.create_load_balancer_listeners(self.name, complex_listeners=complex_listeners)
    balancers = self.conn.get_all_load_balancers(load_balancer_names=[self.name])
    self.assertEqual([lb.name for lb in balancers], [self.name])
    self.assertEqual(sorted((l.get_complex_tuple() for l in balancers[0].listeners)), sorted([(80, 8000, 'HTTP', 'HTTP')] + complex_listeners))