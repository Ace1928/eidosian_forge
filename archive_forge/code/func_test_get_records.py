import time
from tests.compat import unittest
from nose.plugins.attrib import attr
from boto.route53.connection import Route53Connection
from boto.exception import TooManyRecordsException
from boto.vpc import VPCConnection
def test_get_records(self):
    self.zone.get_records()