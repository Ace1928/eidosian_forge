from __future__ import (absolute_import, division, print_function)
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.community.zabbix.plugins.module_utils.base import ZabbixBase
import ansible_collections.community.zabbix.plugins.module_utils.helpers as zabbix_utils
def is_part_of(self, inp, out):
    verdict = True
    for rule, value in inp.items():
        if not isinstance(value, list):
            verdict = verdict and self.__find_val(out.get(rule, ''), value)
        elif len(value):
            if not isinstance(value[0], dict):
                verdict = verdict and self.__find_list(out.get(rule, []), value)
            else:
                for v in value:
                    verdict = verdict and self.__find_dict(out.get(rule, {}), v)
        else:
            verdict = verdict and self.__find_list(rule, value)
    return verdict