import unittest
from datetime import datetime
from mock import Mock
from tests.unit import AWSMockServiceTestCase
from boto.cloudformation.connection import CloudFormationConnection
from boto.exception import BotoServerError
from boto.compat import json
def test_describe_stack_resource_fails(self):
    self.set_http_response(status_code=400)
    with self.assertRaises(self.service_connection.ResponseError):
        api_response = self.service_connection.describe_stack_resource('stack_name', 'resource_id')