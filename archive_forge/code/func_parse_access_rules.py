from __future__ import absolute_import, division, print_function
from copy import deepcopy
from re import M, findall, search
from ansible_collections.ansible.netcommon.plugins.module_utils.network.common import utils
from ansible_collections.vyos.vyos.plugins.module_utils.network.vyos.argspec.firewall_interfaces.firewall_interfaces import (
def parse_access_rules(self, conf):
    """
        This function forms the regex to fetch the 'access-rules'
        for specific interface.
        :param conf: configuration data.
        :return: generated access-rules list configuration.
        """
    ar_lst = []
    v4_ar = findall('^.*(in|out|local) name .*$', conf, M)
    v6_ar = findall('^.*(in|out|local) ipv6-name .*$', conf, M)
    if v4_ar:
        v4_conf = '\n'.join(findall('(^.*?%s.*?$)' % ' name', conf, M))
        config = self.parse_int_rules(v4_conf, 'ipv4')
        if config:
            ar_lst.append(config)
    if v6_ar:
        v6_conf = '\n'.join(findall('(^.*?%s.*?$)' % ' ipv6-name', conf, M))
        config = self.parse_int_rules(v6_conf, 'ipv6')
        if config:
            ar_lst.append(config)
    if ar_lst:
        ar_lst = sorted(ar_lst, key=lambda i: i['afi'])
    else:
        empty_rules = findall('^.*(in|out|local).*', conf, M)
        if empty_rules:
            ar_lst.append({'afi': 'ipv4', 'rules': []})
            ar_lst.append({'afi': 'ipv6', 'rules': []})
    return ar_lst