import boto
import time
from tests.compat import unittest
from boto.ec2.elb import ELBConnection
import boto.ec2.elb
def test_load_balancer_get_attributes(self):
    attributes = self.balancer.get_attributes()
    connection_draining = self.conn.get_lb_attribute(self.balancer.name, 'ConnectionDraining')
    self.assertEqual(connection_draining.enabled, attributes.connection_draining.enabled)
    self.assertEqual(connection_draining.timeout, attributes.connection_draining.timeout)
    access_log = self.conn.get_lb_attribute(self.balancer.name, 'AccessLog')
    self.assertEqual(access_log.enabled, attributes.access_log.enabled)
    self.assertEqual(access_log.s3_bucket_name, attributes.access_log.s3_bucket_name)
    self.assertEqual(access_log.s3_bucket_prefix, attributes.access_log.s3_bucket_prefix)
    self.assertEqual(access_log.emit_interval, attributes.access_log.emit_interval)
    cross_zone_load_balancing = self.conn.get_lb_attribute(self.balancer.name, 'CrossZoneLoadBalancing')
    self.assertEqual(cross_zone_load_balancing, attributes.cross_zone_load_balancing.enabled)