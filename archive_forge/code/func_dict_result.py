import json
import time
from tempest.lib.cli import base
from tempest.lib.common.utils import data_utils
from manilaclient import config
def dict_result(self, object_name, command, client=None):
    """Returns output for the given command as dictionary"""
    output = self.openstack(object_name, params=command, client=client)
    result_dict = self._get_property_from_output(output)
    return result_dict