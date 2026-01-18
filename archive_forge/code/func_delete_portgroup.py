import configparser
import os
from tempest.lib.cli import base
from tempest.lib.common.utils import data_utils
from tempest.lib import exceptions
import ironicclient.tests.functional.utils as utils
def delete_portgroup(self, portgroup_id, ignore_exceptions=False):
    """Delete a port group."""
    try:
        self.ironic('portgroup-delete', flags=self.pg_api_ver, params=portgroup_id)
    except exceptions.CommandFailed:
        if not ignore_exceptions:
            raise