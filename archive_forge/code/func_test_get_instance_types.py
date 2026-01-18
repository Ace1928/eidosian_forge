from tests.unit import unittest
from tests.unit import AWSMockServiceTestCase
import boto.ec2
from boto.ec2.connection import EC2Connection
def test_get_instance_types(self):
    self.set_http_response(status_code=200)
    response = self.ec2.get_all_instance_types()
    self.assertEqual(len(response), 18)
    instance_type = response[0]
    self.assertEqual(instance_type.name, 'm1.small')
    self.assertEqual(instance_type.cores, '1')
    self.assertEqual(instance_type.disk, '5')
    self.assertEqual(instance_type.memory, '256')
    instance_type = response[17]
    self.assertEqual(instance_type.name, 'hs1.8xlarge')
    self.assertEqual(instance_type.cores, '48')
    self.assertEqual(instance_type.disk, '24000')
    self.assertEqual(instance_type.memory, '119808')