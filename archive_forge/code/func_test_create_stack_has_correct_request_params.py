import unittest
from datetime import datetime
from mock import Mock
from tests.unit import AWSMockServiceTestCase
from boto.cloudformation.connection import CloudFormationConnection
from boto.exception import BotoServerError
from boto.compat import json
def test_create_stack_has_correct_request_params(self):
    self.set_http_response(status_code=200)
    api_response = self.service_connection.create_stack('stack_name', template_url='http://url', template_body=SAMPLE_TEMPLATE, parameters=[('KeyName', 'myKeyName')], tags={'TagKey': 'TagValue'}, notification_arns=['arn:notify1', 'arn:notify2'], disable_rollback=True, timeout_in_minutes=20, capabilities=['CAPABILITY_IAM'])
    self.assertEqual(api_response, self.stack_id)
    self.assert_request_parameters({'Action': 'CreateStack', 'Capabilities.member.1': 'CAPABILITY_IAM', 'ContentType': 'JSON', 'DisableRollback': 'true', 'NotificationARNs.member.1': 'arn:notify1', 'NotificationARNs.member.2': 'arn:notify2', 'Parameters.member.1.ParameterKey': 'KeyName', 'Parameters.member.1.ParameterValue': 'myKeyName', 'Tags.member.1.Key': 'TagKey', 'Tags.member.1.Value': 'TagValue', 'StackName': 'stack_name', 'Version': '2010-05-15', 'TimeoutInMinutes': 20, 'TemplateBody': SAMPLE_TEMPLATE, 'TemplateURL': 'http://url'})