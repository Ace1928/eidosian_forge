import configparser
import os
from tempest.lib.cli import base
from tempest.lib.common.utils import data_utils
from tempest.lib import exceptions
import ironicclient.tests.functional.utils as utils
def get_uuids_from_port_list(self):
    port_list = self.list_ports()
    return [x['UUID'] for x in port_list]