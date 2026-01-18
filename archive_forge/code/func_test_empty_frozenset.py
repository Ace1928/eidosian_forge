import datetime
import itertools
from xmlrpc import client as xmlrpclib
import netaddr
from oslotest import base as test_base
from oslo_serialization import msgpackutils
from oslo_utils import uuidutils
def test_empty_frozenset(self):
    self.assertEqual(frozenset([]), _dumps_loads(frozenset([])))