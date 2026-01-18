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
def paged_body(self):
    return b'\n<ListResourceRecordSetsResponse xmlns="https://route53.amazonaws.com/doc/2013-04-01/">\n  <ResourceRecordSets>\n    <ResourceRecordSet>\n      <Name>wrr.example.com.</Name>\n      <Type>A</Type>\n      <SetIdentifier>secondary</SetIdentifier>\n      <Weight>50</Weight>\n      <TTL>300</TTL>\n      <ResourceRecords>\n        <ResourceRecord><Value>127.0.0.2</Value></ResourceRecord>\n      </ResourceRecords>\n    </ResourceRecordSet>\n  </ResourceRecordSets>\n  <IsTruncated>false</IsTruncated>\n  <MaxItems>3</MaxItems>\n</ListResourceRecordSetsResponse>'