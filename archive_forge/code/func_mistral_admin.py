import os
import time
from tempest.lib import exceptions
from mistralclient.tests.functional.cli import base
def mistral_admin(self, cmd, params=''):
    self.clients = self._get_admin_clients()
    return self.parser.listing(self.mistral('{0}'.format(cmd), params='{0}'.format(params)))