from __future__ import absolute_import, division, print_function
from copy import deepcopy
from re import M, findall, search
from ansible_collections.ansible.netcommon.plugins.module_utils.network.common import utils
from ansible_collections.vyos.vyos.plugins.module_utils.network.vyos.argspec.ospfv2.ospfv2 import (
def parse_max_metric(self, conf):
    """
        This function triggers the parsing of 'max_metric' attributes
        :param conf: configuration
        :return: generated config dictionary
        """
    cfg_dict = {}
    cfg_dict['router_lsa'] = self.parse_attrib(conf, 'router_lsa', match='router-lsa')
    return cfg_dict