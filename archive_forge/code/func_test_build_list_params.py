import boto
import time
from tests.compat import unittest
from boto.ec2.elb import ELBConnection
import boto.ec2.elb
def test_build_list_params(self):
    params = {}
    self.conn.build_list_params(params, ['thing1', 'thing2', 'thing3'], 'ThingName%d')
    expected_params = {'ThingName1': 'thing1', 'ThingName2': 'thing2', 'ThingName3': 'thing3'}
    self.assertEqual(params, expected_params)