from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.community.zabbix.plugins.module_utils.base import ZabbixBase
import ansible_collections.community.zabbix.plugins.module_utils.helpers as zabbix_utils
def get_host_macro(self, macro_name, host_id):
    try:
        host_macro_list = self._zapi.usermacro.get({'output': 'extend', 'selectSteps': 'extend', 'hostids': [host_id], 'filter': {'macro': macro_name}})
        if len(host_macro_list) > 0:
            return host_macro_list[0]
        return None
    except Exception as e:
        self._module.fail_json(msg='Failed to get host macro %s: %s' % (macro_name, e))