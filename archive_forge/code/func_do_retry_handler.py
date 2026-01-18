from tests.compat import mock
import re
import xml.dom.minidom
from boto.exception import BotoServerError
from boto.route53.connection import Route53Connection
from boto.route53.exception import DNSServerError
from boto.route53.healthcheck import HealthCheck
from boto.route53.record import ResourceRecordSets, Record
from boto.route53.zone import Zone
from nose.plugins.attrib import attr
from tests.unit import AWSMockServiceTestCase
from boto.compat import six
@mock.patch('time.sleep')
def do_retry_handler(self, sleep_mock):

    def incr_retry_handler(func):

        def _wrapper(*args, **kwargs):
            self.calls['count'] += 1
            return func(*args, **kwargs)
        return _wrapper
    orig_retry = self.service_connection._retry_handler
    self.service_connection._retry_handler = incr_retry_handler(orig_retry)
    self.assertEqual(self.calls['count'], 0)
    with self.assertRaises(BotoServerError):
        self.service_connection.get_all_hosted_zones()
    self.assertEqual(self.calls['count'], 7)
    self.service_connection._retry_handler = orig_retry