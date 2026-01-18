import json
import time
from tempest.lib.cli import base
from tempest.lib.common.utils import data_utils
from manilaclient import config
def listing_result(self, object_name, command, client=None):
    """Returns output for the given command as list of dictionaries"""
    output = self.openstack(object_name, params=command, client=client)
    result = self.parser.listing(output)
    return result