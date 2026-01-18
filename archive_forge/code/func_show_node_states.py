import configparser
import os
from tempest.lib.cli import base
from tempest.lib.common.utils import data_utils
from tempest.lib import exceptions
import ironicclient.tests.functional.utils as utils
def show_node_states(self, node_id):
    show_node_states = self.ironic('node-show-states', params=node_id)
    return utils.get_dict_from_output(show_node_states)