import configparser
import os
from tempest.lib.cli import base
from tempest.lib.common.utils import data_utils
from tempest.lib import exceptions
import ironicclient.tests.functional.utils as utils
def get_nodes_uuids_from_node_list(self):
    node_list = self.list_nodes()
    return [x['UUID'] for x in node_list]