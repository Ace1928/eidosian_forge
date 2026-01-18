from __future__ import absolute_import, division, print_function
from copy import deepcopy
from re import M, findall, search
from ansible_collections.ansible.netcommon.plugins.module_utils.network.common import utils
from ansible_collections.vyos.vyos.plugins.module_utils.network.vyos.argspec.ospfv2.ospfv2 import (
def parse_refresh(self, conf, attrib=None):
    """
        This function triggers the parsing of 'refresh' attributes
        :param conf: configuration
        :param attrib: 'refresh'
        :return: generated config dictionary
        """
    cfg_dict = self.parse_attr(conf, ['timers'], match=attrib)
    return cfg_dict