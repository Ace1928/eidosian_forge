import os
import fixtures
from openstackclient.tests.functional import base
def test_networking_commands_help(self):
    """Check networking related commands in help message."""
    raw_output = self.openstack('help network list')
    self.assertIn('List networks', raw_output)
    raw_output = self.openstack('network create --help')
    self.assertIn('Create new network', raw_output)