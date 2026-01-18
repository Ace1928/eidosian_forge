from tempest.lib.exceptions import CommandFailed
from designateclient.functionaltests.base import BaseDesignateTest
from designateclient.functionaltests.datagen import random_blacklist
from designateclient.functionaltests.v2.fixtures import BlacklistFixture
def test_invalid_blacklist_command(self):
    client = self.clients.as_user('admin')
    cmd = 'zone blacklist notacommand'
    self.assertRaises(CommandFailed, client.openstack, cmd)