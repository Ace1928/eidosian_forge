import os
import time
import os_client_config
from oslo_utils import uuidutils
from tempest.lib.cli import base
from tempest.lib import exceptions
def aodh(self, *args, **kwargs):
    return self.clients.aodh(*args, **kwargs)