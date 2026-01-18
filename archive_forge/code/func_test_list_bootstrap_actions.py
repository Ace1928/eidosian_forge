import boto.utils
from datetime import datetime
from time import time
from tests.unit import AWSMockServiceTestCase
from boto.compat import six
from boto.emr.connection import EmrConnection
from boto.emr.emrobject import BootstrapAction, BootstrapActionList, \
def test_list_bootstrap_actions(self):
    self.set_http_response(200)
    with self.assertRaises(TypeError):
        self.service_connection.list_bootstrap_actions()
    response = self.service_connection.list_bootstrap_actions(cluster_id='j-123')
    self.assert_request_parameters({'Action': 'ListBootstrapActions', 'ClusterId': 'j-123', 'Version': '2009-03-31'})