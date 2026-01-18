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
def test_get_zone(self):
    self.set_http_response(status_code=201)
    response = self.service_connection.get_zone('example.com.')
    self.assertTrue(isinstance(response, Zone))
    self.assertEqual(response.name, 'example.com.')