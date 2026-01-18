from __future__ import absolute_import, division, print_function
import re
from copy import deepcopy
from re import M, findall, search
from ansible_collections.ansible.netcommon.plugins.module_utils.network.common import utils
from ansible_collections.vyos.vyos.plugins.module_utils.network.vyos.argspec.firewall_rules.firewall_rules import (
def parse_rules_lst(self, conf):
    """
        This function forms the regex to fetch the 'rules' with in
        'rule-sets'
        :param conf: configuration data.
        :return: generated rule list configuration.
        """
    r_lst = []
    rules = findall("rule (?:\\'*)(\\d+)(?:\\'*)", conf, M)
    if rules:
        rules_lst = []
        for r in set(rules):
            r_regex = ' %s .+$' % r
            cfg = '\n'.join(findall(r_regex, conf, M))
            obj = self.parse_rules(cfg)
            obj['number'] = int(r)
            if obj:
                rules_lst.append(obj)
        r_lst = sorted(rules_lst, key=lambda i: i['number'])
    return r_lst