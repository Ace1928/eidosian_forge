import datetime
import itertools
from xmlrpc import client as xmlrpclib
import netaddr
from oslotest import base as test_base
from oslo_serialization import msgpackutils
from oslo_utils import uuidutils
def test_bad_register(self):
    registry = msgpackutils.default_registry
    self.assertRaises(ValueError, registry.register, MySpecialSetHandler(), reserved=True, override=True)
    self.assertRaises(ValueError, registry.register, MySpecialSetHandler())
    registry = registry.copy(unfreeze=True)
    registry.register(ColorHandler())
    self.assertRaises(ValueError, registry.register, ColorHandler())