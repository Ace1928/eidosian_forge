import boto
import time
from tests.compat import unittest
from boto.ec2.elb import ELBConnection
import boto.ec2.elb
def test_load_balancer_access_log(self):
    attributes = self.balancer.get_attributes()
    self.assertEqual(False, attributes.access_log.enabled)
    attributes.access_log.enabled = True
    attributes.access_log.s3_bucket_name = self.bucket_name
    attributes.access_log.s3_bucket_prefix = 'access-logs'
    attributes.access_log.emit_interval = 5
    self.conn.modify_lb_attribute(self.balancer.name, 'accessLog', attributes.access_log)
    new_attributes = self.balancer.get_attributes()
    self.assertEqual(True, new_attributes.access_log.enabled)
    self.assertEqual(self.bucket_name, new_attributes.access_log.s3_bucket_name)
    self.assertEqual('access-logs', new_attributes.access_log.s3_bucket_prefix)
    self.assertEqual(5, new_attributes.access_log.emit_interval)