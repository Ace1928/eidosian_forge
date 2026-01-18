import boto
import time
from tests.compat import unittest
from boto.ec2.elb import ELBConnection
import boto.ec2.elb
def change_and_verify_load_balancer_connection_draining(self, enabled, timeout=None):
    attributes = self.balancer.get_attributes()
    attributes.connection_draining.enabled = enabled
    if timeout is not None:
        attributes.connection_draining.timeout = timeout
    self.conn.modify_lb_attribute(self.balancer.name, 'ConnectionDraining', attributes.connection_draining)
    attributes = self.balancer.get_attributes()
    self.assertEqual(enabled, attributes.connection_draining.enabled)
    if timeout is not None:
        self.assertEqual(timeout, attributes.connection_draining.timeout)