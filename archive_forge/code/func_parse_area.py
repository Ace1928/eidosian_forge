from __future__ import absolute_import, division, print_function
from copy import deepcopy
from re import M, findall, search
from ansible_collections.ansible.netcommon.plugins.module_utils.network.common import utils
from ansible_collections.vyos.vyos.plugins.module_utils.network.vyos.argspec.ospfv2.ospfv2 import (
def parse_area(self, conf, area_id):
    """
        This function triggers the parsing of 'area' attributes.
        :param conf: configuration data
        :param area_id: area identity
        :return: generated rule configuration dictionary.
        """
    rule = self.parse_attrib(conf, 'area_id', match=area_id)
    r_sub = {'area_type': self.parse_area_type(conf, 'area-type'), 'network': self.parse_network(conf), 'range': self.parse_attrib_list(conf, 'range', 'address'), 'virtual_link': self.parse_attrib_list(conf, 'virtual-link', 'address')}
    rule.update(r_sub)
    return rule