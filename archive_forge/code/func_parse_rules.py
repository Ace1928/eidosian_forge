from __future__ import absolute_import, division, print_function
from copy import deepcopy
from re import M, findall, search
from ansible_collections.ansible.netcommon.plugins.module_utils.network.common import utils
from ansible_collections.vyos.vyos.plugins.module_utils.network.vyos.argspec.firewall_interfaces.firewall_interfaces import (
def parse_rules(self, conf, afi):
    """
        This function triggers the parsing of 'rule' attributes.
        a_lst is a list having rule attributes which doesn't
        have further sub attributes.
        :param conf: configuration.
        :param afi: ip address type.
        :return: generated rule configuration dictionary.
        """
    cfg = {}
    out = findall('[^\\s]+', conf, M)
    if out:
        cfg['direction'] = out[0].strip("'")
        if afi == 'ipv6':
            out = findall("[^\\s]+ ipv6-name (?:\\'*)(\\S+)(?:\\'*)", conf, M)
            if out:
                cfg['name'] = str(out[0]).strip("'")
        else:
            out = findall("[^\\s]+ name (?:\\'*)(\\S+)(?:\\'*)", conf, M)
            if out:
                cfg['name'] = out[-1].strip("'")
    return cfg