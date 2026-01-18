import boto.utils
from datetime import datetime
from time import time
from tests.unit import AWSMockServiceTestCase
from boto.compat import six
from boto.emr.connection import EmrConnection
from boto.emr.emrobject import BootstrapAction, BootstrapActionList, \
def test_describe_step(self):
    self.set_http_response(200)
    with self.assertRaises(TypeError):
        self.service_connection.describe_step()
    with self.assertRaises(TypeError):
        self.service_connection.describe_step(cluster_id='j-123')
    with self.assertRaises(TypeError):
        self.service_connection.describe_step(step_id='abc')
    response = self.service_connection.describe_step(cluster_id='j-123', step_id='abc')
    self.assert_request_parameters({'Action': 'DescribeStep', 'ClusterId': 'j-123', 'StepId': 'abc', 'Version': '2009-03-31'})