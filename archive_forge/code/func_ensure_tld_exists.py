from tempest.lib.cli import base
from tempest.lib.exceptions import CommandFailed
from designateclient.functionaltests import client
from designateclient.functionaltests import config
def ensure_tld_exists(self, tld):
    try:
        self.clients.as_user('admin').tld_create(tld)
    except CommandFailed:
        pass