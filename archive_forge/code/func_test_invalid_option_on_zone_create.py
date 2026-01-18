from tempest.lib.exceptions import CommandFailed
from designateclient.functionaltests.base import BaseDesignateTest
from designateclient.functionaltests.datagen import random_zone_name
from designateclient.functionaltests.v2.fixtures import ZoneFixture
def test_invalid_option_on_zone_create(self):
    cmd = 'zone create {} --invalid "not a valid option"'.format(random_zone_name())
    self.assertRaises(CommandFailed, self.clients.openstack, cmd)