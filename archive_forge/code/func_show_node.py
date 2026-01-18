import configparser
import os
from tempest.lib.cli import base
from tempest.lib.common.utils import data_utils
from tempest.lib import exceptions
import ironicclient.tests.functional.utils as utils
def show_node(self, node_id, params=''):
    node_show = self.ironic('node-show', params='{0} {1}'.format(node_id, params))
    return utils.get_dict_from_output(node_show)