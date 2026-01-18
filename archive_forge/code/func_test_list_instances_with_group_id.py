import boto.utils
from datetime import datetime
from time import time
from tests.unit import AWSMockServiceTestCase
from boto.compat import six
from boto.emr.connection import EmrConnection
from boto.emr.emrobject import BootstrapAction, BootstrapActionList, \
def test_list_instances_with_group_id(self):
    self.set_http_response(200)
    response = self.service_connection.list_instances(cluster_id='j-123', instance_group_id='abc')
    self.assert_request_parameters({'Action': 'ListInstances', 'ClusterId': 'j-123', 'InstanceGroupId': 'abc', 'Version': '2009-03-31'})