from __future__ import absolute_import, division, print_function
from copy import deepcopy
from re import M, findall, search
from ansible_collections.ansible.netcommon.plugins.module_utils.network.common import utils
from ansible_collections.vyos.vyos.plugins.module_utils.network.vyos.argspec.firewall_global.firewall_global import (
def parse_address_port_lst(self, conf, name, key):
    """
        This function forms the regex to fetch the
        group members attributes.
        :param conf: configuration data.
        :param name: name of group.
        :param key: key value.
        :return: generated member list configuration.
        """
    l_lst = []
    attribs = findall('^.*' + name + ' ' + key + ' (\\S+)', conf, M)
    if attribs:
        for attr in attribs:
            if key == 'port':
                l_lst.append({'port': attr.strip("'")})
            else:
                l_lst.append({'address': attr.strip("'")})
    return l_lst