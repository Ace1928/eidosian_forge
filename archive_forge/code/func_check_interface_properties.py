from __future__ import absolute_import, division, print_function
import copy
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.community.zabbix.plugins.module_utils.base import ZabbixBase
import ansible_collections.community.zabbix.plugins.module_utils.helpers as zabbix_utils
def check_interface_properties(self, exist_interfaces, interfaces):
    if len(exist_interfaces) != len(interfaces):
        return True
    for iface in interfaces:
        found = False
        for e_int in exist_interfaces:
            diff_dict = {}
            zabbix_utils.helper_cleanup_data(zabbix_utils.helper_compare_dictionaries(iface, e_int, diff_dict))
            if diff_dict == {}:
                found = True
                break
    if interfaces and (not found):
        return True
    return False