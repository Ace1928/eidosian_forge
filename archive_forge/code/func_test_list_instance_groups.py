import boto.utils
from datetime import datetime
from time import time
from tests.unit import AWSMockServiceTestCase
from boto.compat import six
from boto.emr.connection import EmrConnection
from boto.emr.emrobject import BootstrapAction, BootstrapActionList, \
def test_list_instance_groups(self):
    self.set_http_response(200)
    with self.assertRaises(TypeError):
        self.service_connection.list_instance_groups()
    response = self.service_connection.list_instance_groups(cluster_id='j-123')
    self.assert_request_parameters({'Action': 'ListInstanceGroups', 'ClusterId': 'j-123', 'Version': '2009-03-31'})
    self.assertTrue(isinstance(response, InstanceGroupList))
    self.assertEqual(len(response.instancegroups), 2)
    self.assertTrue(isinstance(response.instancegroups[0], InstanceGroupInfo))
    self.assertEqual(response.instancegroups[0].id, 'ig-aaaaaaaaaaaaa')
    self.assertEqual(response.instancegroups[0].instancegrouptype, 'MASTER')
    self.assertEqual(response.instancegroups[0].instancetype, 'm1.large')
    self.assertEqual(response.instancegroups[0].market, 'ON_DEMAND')
    self.assertEqual(response.instancegroups[0].name, 'Master instance group')
    self.assertEqual(response.instancegroups[0].requestedinstancecount, '1')
    self.assertEqual(response.instancegroups[0].runninginstancecount, '0')
    self.assertTrue(isinstance(response.instancegroups[0].status, ClusterStatus))
    self.assertEqual(response.instancegroups[0].status.state, 'TERMINATED')