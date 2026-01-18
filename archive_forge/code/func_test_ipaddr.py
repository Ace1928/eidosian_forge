import datetime
import itertools
from xmlrpc import client as xmlrpclib
import netaddr
from oslotest import base as test_base
from oslo_serialization import msgpackutils
from oslo_utils import uuidutils
def test_ipaddr(self):
    thing = {'ip_addr': netaddr.IPAddress('1.2.3.4')}
    self.assertEqual(thing, _dumps_loads(thing))