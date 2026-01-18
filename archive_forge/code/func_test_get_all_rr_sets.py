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
def test_get_all_rr_sets(self):
    self.set_http_response(status_code=200)
    response = self.service_connection.get_all_rrsets('Z1111', maxitems=3)
    self.assertEqual(self.actual_request.path, '/2013-04-01/hostedzone/Z1111/rrset?maxitems=3')
    self.set_http_response(status_code=200, body=self.paged_body())
    self.assertEqual(len(list(response)), 4)
    url_parts = urllib.parse.urlparse(self.actual_request.path)
    self.assertEqual(url_parts.path, '/2013-04-01/hostedzone/Z1111/rrset')
    self.assertEqual(urllib.parse.parse_qs(url_parts.query), dict(type=['A'], name=['wrr.example.com.'], identifier=['secondary']))