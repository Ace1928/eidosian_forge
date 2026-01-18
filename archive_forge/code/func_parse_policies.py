from __future__ import absolute_import, division, print_function
from copy import deepcopy
from re import M, findall, search
from ansible_collections.ansible.netcommon.plugins.module_utils.network.common import utils
from ansible_collections.vyos.vyos.plugins.module_utils.network.vyos.argspec.firewall_global.firewall_global import (
def parse_policies(self, conf, attrib=None):
    """
        This function triggers the parsing of policy attributes
        action and log.
        :param conf: configuration
        :param attrib: connection type.
        :return: generated rule configuration dictionary.
        """
    a_lst = ['action', 'log']
    cfg_dict = self.parse_attr(conf, a_lst, match=attrib)
    return cfg_dict