from __future__ import absolute_import, division, print_function
from copy import deepcopy
from re import M, findall, search
from ansible_collections.ansible.netcommon.plugins.module_utils.network.common import utils
from ansible_collections.vyos.vyos.plugins.module_utils.network.vyos.argspec.firewall_global.firewall_global import (
def parse_icmp_redirects(self, conf, attrib=None):
    """
        This function triggers the parsing of 'icmp_redirects' attributes.
        :param conf: configuration to be parsed.
        :param attrib: 'ipv4/ipv6'.
        :return: generated config dictionary.
        """
    a_lst = ['send', 'receive']
    cfg_dict = self.parse_attr(conf, a_lst, type=attrib)
    return cfg_dict