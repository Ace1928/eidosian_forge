import json
from tests.unit import unittest
from tests.unit import AWSMockServiceTestCase
from mock import Mock
from boto.sns.connection import SNSConnection
def test_create_platform_endpoint(self):
    self.set_http_response(status_code=200)
    self.service_connection.create_platform_endpoint(platform_application_arn='arn:myapp', token='abcde12345', custom_user_data='john', attributes={'Enabled': False})
    self.assert_request_parameters({'Action': 'CreatePlatformEndpoint', 'PlatformApplicationArn': 'arn:myapp', 'Token': 'abcde12345', 'CustomUserData': 'john', 'Attributes.entry.1.key': 'Enabled', 'Attributes.entry.1.value': False}, ignore_params_values=['Version', 'ContentType'])