from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.community.zabbix.plugins.module_utils.base import ZabbixBase
import ansible_collections.community.zabbix.plugins.module_utils.helpers as zabbix_utils
def get_global_macro(self, macro_name):
    try:
        all_global_macro_list = self._zapi.usermacro.get({'globalmacro': 'true'})
        global_macro_list = [d for d in all_global_macro_list if d['macro'] == macro_name]
        if len(global_macro_list) > 0:
            return global_macro_list[0]
        return None
    except Exception as e:
        self._module.fail_json(msg='Failed to get global macro %s: %s' % (macro_name, e))