import boto
import time
from tests.compat import unittest
from boto.ec2.elb import ELBConnection
import boto.ec2.elb
def test_can_make_sigv4_call(self):
    connection = boto.ec2.elb.connect_to_region('eu-central-1')
    lbs = connection.get_all_load_balancers()
    self.assertTrue(isinstance(lbs, list))