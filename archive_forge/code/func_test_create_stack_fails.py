import unittest
from datetime import datetime
from mock import Mock
from tests.unit import AWSMockServiceTestCase
from boto.cloudformation.connection import CloudFormationConnection
from boto.exception import BotoServerError
from boto.compat import json
def test_create_stack_fails(self):
    self.set_http_response(status_code=400, reason='Bad Request', body=b'{"Error": {"Code": 1, "Message": "Invalid arg."}}')
    with self.assertRaisesRegexp(self.service_connection.ResponseError, 'Invalid arg.'):
        api_response = self.service_connection.create_stack('stack_name', template_body=SAMPLE_TEMPLATE, parameters=[('KeyName', 'myKeyName')])