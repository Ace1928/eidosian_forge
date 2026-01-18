import time
import uuid
from tempest.lib import exceptions
from openstackclient.tests.functional import base
def server_delete(self, name):
    """Delete server by name"""
    raw_output = self.openstack('server delete ' + name)
    self.assertOutput('', raw_output)