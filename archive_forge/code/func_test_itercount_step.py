import datetime
import itertools
from xmlrpc import client as xmlrpclib
import netaddr
from oslotest import base as test_base
from oslo_serialization import msgpackutils
from oslo_utils import uuidutils
def test_itercount_step(self):
    it = itertools.count(1, 3)
    it2 = _dumps_loads(it)
    self.assertEqual(next(it), next(it2))