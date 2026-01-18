import configparser
import os
from tempest.lib.cli import base
from tempest.lib.common.utils import data_utils
from tempest.lib import exceptions
import ironicclient.tests.functional.utils as utils
def show_port(self, port_id, params=''):
    port_show = self.ironic('port-show', params='{0} {1}'.format(port_id, params))
    return utils.get_dict_from_output(port_show)