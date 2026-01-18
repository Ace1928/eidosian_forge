import os
import time
from tempest.lib import exceptions
from mistralclient.tests.functional.cli import base
def mistral_cli(self, admin, cmd, params=''):
    if admin:
        return self.mistral_admin(cmd, params)
    else:
        return self.mistral_alt_user(cmd, params)