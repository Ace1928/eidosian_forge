import boto.utils
from datetime import datetime
from time import time
from tests.unit import AWSMockServiceTestCase
from boto.compat import six
from boto.emr.connection import EmrConnection
from boto.emr.emrobject import BootstrapAction, BootstrapActionList, \
def test_list_clusters(self):
    self.set_http_response(status_code=200)
    response = self.service_connection.list_clusters()
    self.assert_request_parameters({'Action': 'ListClusters', 'Version': '2009-03-31'})
    self.assertTrue(isinstance(response, ClusterSummaryList))
    self.assertEqual(len(response.clusters), 2)
    self.assertTrue(isinstance(response.clusters[0], ClusterSummary))
    self.assertEqual(response.clusters[0].name, 'analytics test')
    self.assertEqual(response.clusters[0].normalizedinstancehours, '10')
    self.assertTrue(isinstance(response.clusters[0].status, ClusterStatus))
    self.assertEqual(response.clusters[0].status.state, 'TERMINATED')
    self.assertTrue(isinstance(response.clusters[0].status.timeline, ClusterTimeline))
    self.assertEqual(response.clusters[0].status.timeline.creationdatetime, '2014-01-24T01:21:21Z')
    self.assertEqual(response.clusters[0].status.timeline.readydatetime, '2014-01-24T01:25:26Z')
    self.assertEqual(response.clusters[0].status.timeline.enddatetime, '2014-01-24T02:19:46Z')
    self.assertTrue(isinstance(response.clusters[0].status.statechangereason, ClusterStateChangeReason))
    self.assertEqual(response.clusters[0].status.statechangereason.code, 'USER_REQUEST')
    self.assertEqual(response.clusters[0].status.statechangereason.message, 'Terminated by user request')