import unittest
import time
import telnetlib
import socket
from nose.plugins.attrib import attr
from boto.ec2.connection import EC2Connection
from boto.exception import EC2ResponseError
import boto.ec2
def test_can_get_all_instances_sigv4(self):
    connection = boto.ec2.connect_to_region('eu-central-1')
    self.assertTrue(isinstance(connection.get_all_instances(), list))