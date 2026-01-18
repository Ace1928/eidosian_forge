import boto.utils
from datetime import datetime
from time import time
from tests.unit import AWSMockServiceTestCase
from boto.compat import six
from boto.emr.connection import EmrConnection
from boto.emr.emrobject import BootstrapAction, BootstrapActionList, \
def test_describe_jobflows_no_args(self):
    self.set_http_response(200)
    self.service_connection.describe_jobflows()
    self.assert_request_parameters({'Action': 'DescribeJobFlows'}, ignore_params_values=['Version'])