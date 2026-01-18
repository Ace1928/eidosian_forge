import configparser
import os
from tempest.lib.cli import base
from tempest.lib.common.utils import data_utils
from tempest.lib import exceptions
import ironicclient.tests.functional.utils as utils
def list_node_chassis(self, chassis_uuid, params=''):
    return self.ironic('chassis-node-list', params='{0} {1}'.format(chassis_uuid, params))