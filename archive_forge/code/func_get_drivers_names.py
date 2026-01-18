import configparser
import os
from tempest.lib.cli import base
from tempest.lib.common.utils import data_utils
from tempest.lib import exceptions
import ironicclient.tests.functional.utils as utils
def get_drivers_names(self):
    driver_list = self.list_driver()
    return [x['Supported driver(s)'] for x in driver_list]