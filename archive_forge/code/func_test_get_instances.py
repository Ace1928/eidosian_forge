from tests.compat import unittest
from tests.unit import AWSMockServiceTestCase
from boto.ec2.connection import EC2Connection
from boto.ec2.securitygroup import SecurityGroup
def test_get_instances(self):
    self.set_http_response(status_code=200, body=DESCRIBE_SECURITY_GROUP)
    groups = self.service_connection.get_all_security_groups()
    self.set_http_response(status_code=200, body=DESCRIBE_INSTANCES)
    instances = groups[0].instances()
    self.assertEqual(1, len(instances))
    self.assertEqual(groups[0].id, instances[0].groups[0].id)