import boto.utils
from datetime import datetime
from time import time
from tests.unit import AWSMockServiceTestCase
from boto.compat import six
from boto.emr.connection import EmrConnection
from boto.emr.emrobject import BootstrapAction, BootstrapActionList, \
def test_list_instances_with_types(self):
    self.set_http_response(200)
    response = self.service_connection.list_instances(cluster_id='j-123', instance_group_types=['MASTER', 'TASK'])
    self.assert_request_parameters({'Action': 'ListInstances', 'ClusterId': 'j-123', 'InstanceGroupTypes.member.1': 'MASTER', 'InstanceGroupTypes.member.2': 'TASK', 'Version': '2009-03-31'})