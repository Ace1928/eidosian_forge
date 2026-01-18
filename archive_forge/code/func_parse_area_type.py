from __future__ import absolute_import, division, print_function
from copy import deepcopy
from re import M, findall, search
from ansible_collections.ansible.netcommon.plugins.module_utils.network.common import utils
from ansible_collections.vyos.vyos.plugins.module_utils.network.vyos.argspec.ospfv2.ospfv2 import (
def parse_area_type(self, conf, attrib=None):
    """
        This function triggers the parsing of 'area_type' attributes
        :param conf: configuration
        :param attrib: 'area-type'
        :return: generated config dictionary
        """
    cfg_dict = self.parse_attr(conf, ['normal'], match=attrib)
    cfg_dict['nssa'] = self.parse_attrib(conf, 'nssa', match='nssa')
    cfg_dict['stub'] = self.parse_attrib(conf, 'stub', match='stub')
    return cfg_dict