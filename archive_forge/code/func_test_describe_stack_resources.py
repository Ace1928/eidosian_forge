import unittest
from datetime import datetime
from mock import Mock
from tests.unit import AWSMockServiceTestCase
from boto.cloudformation.connection import CloudFormationConnection
from boto.exception import BotoServerError
from boto.compat import json
def test_describe_stack_resources(self):
    self.set_http_response(status_code=200)
    first, second = self.service_connection.describe_stack_resources('stack_name', 'logical_resource_id', 'physical_resource_id')
    self.assertEqual(first.description, None)
    self.assertEqual(first.logical_resource_id, 'MyDBInstance')
    self.assertEqual(first.physical_resource_id, 'MyStack_DB1')
    self.assertEqual(first.resource_status, 'CREATE_COMPLETE')
    self.assertEqual(first.resource_status_reason, None)
    self.assertEqual(first.resource_type, 'AWS::DBInstance')
    self.assertEqual(first.stack_id, 'arn:aws:cfn:us-east-1:1:stack')
    self.assertEqual(first.stack_name, 'MyStack')
    self.assertIsNotNone(first.timestamp)
    self.assertEqual(second.description, None)
    self.assertEqual(second.logical_resource_id, 'MyAutoScalingGroup')
    self.assertEqual(second.physical_resource_id, 'MyStack_ASG1')
    self.assertEqual(second.resource_status, 'CREATE_IN_PROGRESS')
    self.assertEqual(second.resource_status_reason, None)
    self.assertEqual(second.resource_type, 'AWS::AutoScalingGroup')
    self.assertEqual(second.stack_id, 'arn:aws:cfn:us-east-1:1:stack')
    self.assertEqual(second.stack_name, 'MyStack')
    self.assertIsNotNone(second.timestamp)
    self.assert_request_parameters({'Action': 'DescribeStackResources', 'LogicalResourceId': 'logical_resource_id', 'PhysicalResourceId': 'physical_resource_id', 'StackName': 'stack_name', 'Version': '2010-05-15'})