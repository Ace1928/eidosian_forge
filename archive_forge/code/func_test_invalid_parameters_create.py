from unittest import mock
from novaclient import api_versions
from novaclient import base
from novaclient import exceptions
from novaclient.tests.unit import utils
from novaclient.tests.unit.v2 import fakes
from novaclient.v2 import flavors
def test_invalid_parameters_create(self):
    self.assertRaises(exceptions.CommandError, self.cs.flavors.create, 'flavorcreate', 'invalid', 1, 10, 1234, swap=0, ephemeral=0, rxtx_factor=1.0, is_public=True)
    self.assertRaises(exceptions.CommandError, self.cs.flavors.create, 'flavorcreate', 512, 'invalid', 10, 1234, swap=0, ephemeral=0, rxtx_factor=1.0, is_public=True)
    self.assertRaises(exceptions.CommandError, self.cs.flavors.create, 'flavorcreate', 512, 1, 'invalid', 1234, swap=0, ephemeral=0, rxtx_factor=1.0, is_public=True)
    self.assertRaises(exceptions.CommandError, self.cs.flavors.create, 'flavorcreate', 512, 1, 10, 1234, swap='invalid', ephemeral=0, rxtx_factor=1.0, is_public=True)
    self.assertRaises(exceptions.CommandError, self.cs.flavors.create, 'flavorcreate', 512, 1, 10, 1234, swap=0, ephemeral='invalid', rxtx_factor=1.0, is_public=True)
    self.assertRaises(exceptions.CommandError, self.cs.flavors.create, 'flavorcreate', 512, 1, 10, 1234, swap=0, ephemeral=0, rxtx_factor='invalid', is_public=True)
    self.assertRaises(exceptions.CommandError, self.cs.flavors.create, 'flavorcreate', 512, 1, 10, 1234, swap=0, ephemeral=0, rxtx_factor=1.0, is_public='invalid')