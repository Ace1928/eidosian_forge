from __future__ import absolute_import, division, print_function
from copy import deepcopy
from re import M, findall, search
from ansible_collections.ansible.netcommon.plugins.module_utils.network.common import utils
from ansible_collections.vyos.vyos.plugins.module_utils.network.vyos.argspec.firewall_global.firewall_global import (
def get_src_route(self, attrib):
    """
        This function looks for the attribute in predefined integer type set.
        :param attrib: attribute.
        :return: True/false.
        """
    return 'ipv6_src_route' if attrib == 'ipv6' else 'ip_src_route'