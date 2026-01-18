from __future__ import absolute_import, division, print_function
from copy import deepcopy
from re import M, findall, search
from ansible_collections.ansible.netcommon.plugins.module_utils.network.common import utils
from ansible_collections.vyos.vyos.plugins.module_utils.network.vyos.argspec.firewall_global.firewall_global import (
def parse_state_policy(self, conf):
    """
        This function fetched the connecton type and invoke
        function to parse other state-policy attributes.
        :param conf: configuration data.
        :return: generated rule list configuration.
        """
    sp_lst = []
    attrib = 'state-policy'
    policies = findall('^set firewall ' + attrib + ' (\\S+)', conf, M)
    if policies:
        rules_lst = []
        for sp in set(policies):
            sp_regex = ' %s .+$' % sp
            cfg = '\n'.join(findall(sp_regex, conf, M))
            obj = self.parse_policies(cfg, sp)
            obj['connection_type'] = sp
            if obj:
                rules_lst.append(obj)
        sp_lst = sorted(rules_lst, key=lambda i: i['connection_type'])
    return sp_lst