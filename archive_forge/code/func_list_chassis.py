import configparser
import os
from tempest.lib.cli import base
from tempest.lib.common.utils import data_utils
from tempest.lib import exceptions
import ironicclient.tests.functional.utils as utils
def list_chassis(self, params=''):
    return self.ironic('chassis-list', params=params)