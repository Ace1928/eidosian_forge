import configparser
import os
from tempest.lib.cli import base
from tempest.lib.common.utils import data_utils
from tempest.lib import exceptions
import ironicclient.tests.functional.utils as utils
def get_portgroup_uuids_from_portgroup_list(self):
    """Get UUIDs from list of port groups."""
    portgroup_list = self.list_portgroups()
    return [x['UUID'] for x in portgroup_list]