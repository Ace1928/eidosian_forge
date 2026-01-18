import time
import uuid
from tempest.lib import exceptions
from openstackclient.tests.functional import base
def server_create(self, name=None, cleanup=True):
    """Create server, with cleanup"""
    if not self.flavor_name:
        self.flavor_name = self.get_flavor()
    if not self.image_name:
        self.image_name = self.get_image()
    if not self.network_arg:
        self.network_arg = self.get_network()
    name = name or uuid.uuid4().hex
    cmd_output = self.openstack('server create ' + '--flavor ' + self.flavor_name + ' ' + '--image ' + self.image_name + ' ' + self.network_arg + ' ' + '--wait ' + name, parse_output=True)
    self.assertIsNotNone(cmd_output['id'])
    self.assertEqual(name, cmd_output['name'])
    if cleanup:
        self.addCleanup(self.server_delete, name)
    return cmd_output