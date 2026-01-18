import unittest
from datetime import datetime
from mock import Mock
from tests.unit import AWSMockServiceTestCase
from boto.cloudformation.connection import CloudFormationConnection
from boto.exception import BotoServerError
from boto.compat import json
def test_get_stack_policy(self):
    self.set_http_response(status_code=200)
    api_response = self.service_connection.get_stack_policy('stack-id')
    self.assertEqual(api_response, '{...}')
    self.assert_request_parameters({'Action': 'GetStackPolicy', 'ContentType': 'JSON', 'StackName': 'stack-id', 'Version': '2010-05-15'})