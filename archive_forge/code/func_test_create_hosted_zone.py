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
def test_create_hosted_zone(self):
    self.set_http_response(status_code=201)
    response = self.service_connection.create_hosted_zone('example.com.', 'my_ref', 'a comment')
    self.assertEqual(response['CreateHostedZoneResponse']['DelegationSet']['NameServers'], ['ns-100.awsdns-01.com', 'ns-1000.awsdns-01.co.uk', 'ns-1000.awsdns-01.org', 'ns-900.awsdns-01.net'])
    self.assertEqual(response['CreateHostedZoneResponse']['HostedZone']['Config']['PrivateZone'], u'false')