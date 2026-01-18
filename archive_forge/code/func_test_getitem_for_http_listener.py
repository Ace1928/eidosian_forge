import xml.sax
from tests.unit import unittest
import boto.resultset
from boto.ec2.elb.loadbalancer import LoadBalancer
from boto.ec2.elb.listener import Listener
def test_getitem_for_http_listener(self):
    listener = Listener(load_balancer_port=80, instance_port=80, protocol='HTTP', instance_protocol='HTTP')
    self.assertEqual(listener[0], 80)
    self.assertEqual(listener[1], 80)
    self.assertEqual(listener[2], 'HTTP')
    self.assertEqual(listener[3], 'HTTP')