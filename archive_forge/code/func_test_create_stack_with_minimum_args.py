import unittest
from datetime import datetime
from mock import Mock
from tests.unit import AWSMockServiceTestCase
from boto.cloudformation.connection import CloudFormationConnection
from boto.exception import BotoServerError
from boto.compat import json
def test_create_stack_with_minimum_args(self):
    self.set_http_response(status_code=200)
    api_response = self.service_connection.create_stack('stack_name')
    self.assertEqual(api_response, self.stack_id)
    self.assert_request_parameters({'Action': 'CreateStack', 'ContentType': 'JSON', 'DisableRollback': 'false', 'StackName': 'stack_name', 'Version': '2010-05-15'})