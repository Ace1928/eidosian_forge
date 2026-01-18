import time
from tests.compat import unittest
from nose.plugins.attrib import attr
from boto.route53.connection import Route53Connection
from boto.exception import TooManyRecordsException
from boto.vpc import VPCConnection
def test_identifiers_lbrs(self):
    self.zone.add_a('lbr.%s' % self.base_domain, '4.3.2.1', identifier=('baz', 'us-east-1'))
    self.zone.add_a('lbr.%s' % self.base_domain, '8.7.6.5', identifier=('bam', 'us-west-1'))
    lbrs = self.zone.find_records('lbr.%s' % self.base_domain, 'A', all=True)
    self.assertEquals(len(lbrs), 2)
    self.zone.delete_a('lbr.%s' % self.base_domain, identifier=('bam', 'us-west-1'))
    self.zone.delete_a('lbr.%s' % self.base_domain, identifier=('baz', 'us-east-1'))