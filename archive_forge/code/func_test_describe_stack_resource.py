import unittest
from datetime import datetime
from mock import Mock
from tests.unit import AWSMockServiceTestCase
from boto.cloudformation.connection import CloudFormationConnection
from boto.exception import BotoServerError
from boto.compat import json
def test_describe_stack_resource(self):
    self.set_http_response(status_code=200)
    api_response = self.service_connection.describe_stack_resource('stack_name', 'resource_id')
    self.assertEqual(api_response, 'fake server response')
    self.assert_request_parameters({'Action': 'DescribeStackResource', 'ContentType': 'JSON', 'LogicalResourceId': 'resource_id', 'StackName': 'stack_name', 'Version': '2010-05-15'})