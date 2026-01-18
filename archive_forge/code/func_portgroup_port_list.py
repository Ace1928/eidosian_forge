import configparser
import os
from tempest.lib.cli import base
from tempest.lib.common.utils import data_utils
from tempest.lib import exceptions
import ironicclient.tests.functional.utils as utils
def portgroup_port_list(self, portgroup_id, params=''):
    """List the ports associated with a port group."""
    return self.ironic('portgroup-port-list', flags=self.pg_api_ver, params='{0} {1}'.format(portgroup_id, params))