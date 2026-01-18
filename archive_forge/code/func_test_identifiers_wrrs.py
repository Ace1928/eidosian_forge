import time
from tests.compat import unittest
from nose.plugins.attrib import attr
from boto.route53.connection import Route53Connection
from boto.exception import TooManyRecordsException
from boto.vpc import VPCConnection
def test_identifiers_wrrs(self):
    self.zone.add_a('wrr.%s' % self.base_domain, '1.2.3.4', identifier=('foo', '20'))
    self.zone.add_a('wrr.%s' % self.base_domain, '5.6.7.8', identifier=('bar', '10'))
    wrrs = self.zone.find_records('wrr.%s' % self.base_domain, 'A', all=True)
    self.assertEquals(len(wrrs), 2)
    self.zone.delete_a('wrr.%s' % self.base_domain, all=True)