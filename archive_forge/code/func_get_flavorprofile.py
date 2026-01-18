from octaviaclient.api import constants
from octaviaclient.api.v2 import octavia
from osc_lib import exceptions
from heat.engine.clients import client_plugin
from heat.engine import constraints
def get_flavorprofile(self, value):
    flavorprofile = self.client().find(path=constants.BASE_FLAVORPROFILE_URL, value=value, attr=DEFAULT_FIND_ATTR)
    return flavorprofile['id']