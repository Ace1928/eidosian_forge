from tempest.lib.exceptions import CommandFailed
from designateclient.functionaltests.base import BaseDesignateTest
from designateclient.functionaltests.datagen import random_tld
from designateclient.functionaltests.v2.fixtures import TLDFixture
def test_no_set_tld_with_description_and_no_description(self):
    client = self.clients.as_user('admin')
    self.assertRaises(CommandFailed, client.tld_set, self.tld.id, description='An updated tld', no_description=True)