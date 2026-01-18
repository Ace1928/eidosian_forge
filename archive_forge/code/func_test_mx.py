import time
from tests.compat import unittest
from nose.plugins.attrib import attr
from boto.route53.connection import Route53Connection
from boto.exception import TooManyRecordsException
from boto.vpc import VPCConnection
def test_mx(self):
    self.zone.add_mx(self.base_domain, ['10 mx1.%s' % self.base_domain, '20 mx2.%s' % self.base_domain], 1000)
    record = self.zone.get_mx(self.base_domain)
    self.assertEquals(set(record.resource_records), set([u'10 mx1.%s.' % self.base_domain, u'20 mx2.%s.' % self.base_domain]))
    self.assertEquals(record.ttl, u'1000')
    self.zone.update_mx(self.base_domain, ['10 mail1.%s' % self.base_domain, '20 mail2.%s' % self.base_domain], 50)
    record = self.zone.get_mx(self.base_domain)
    self.assertEquals(set(record.resource_records), set([u'10 mail1.%s.' % self.base_domain, '20 mail2.%s.' % self.base_domain]))
    self.assertEquals(record.ttl, u'50')