import unittest
import six
from apitools.base.py import list_pager
from apitools.base.py.testing import mock
from samples.dns_sample.dns_v1 import dns_v1_client
from samples.dns_sample.dns_v1 import dns_v1_messages
def testRecordSetList(self):
    response_record_set = dns_v1_messages.ResourceRecordSet(kind=u'dns#resourceRecordSet', name=u'zone.com.', rrdatas=[u'1.2.3.4'], ttl=21600, type=u'A')
    self.mocked_dns_v1.resourceRecordSets.List.Expect(dns_v1_messages.DnsResourceRecordSetsListRequest(project=u'my-project', managedZone=u'test_zone_name', type=u'green', maxResults=100), dns_v1_messages.ResourceRecordSetsListResponse(rrsets=[response_record_set]))
    results = list(list_pager.YieldFromList(self.mocked_dns_v1.resourceRecordSets, dns_v1_messages.DnsResourceRecordSetsListRequest(project='my-project', managedZone='test_zone_name', type='green'), limit=100, field='rrsets'))
    self.assertEquals([response_record_set], results)