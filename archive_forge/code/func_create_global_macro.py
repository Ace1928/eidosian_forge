from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.community.zabbix.plugins.module_utils.base import ZabbixBase
import ansible_collections.community.zabbix.plugins.module_utils.helpers as zabbix_utils
def create_global_macro(self, macro_name, macro_value, macro_type, macro_description):
    try:
        if self._module.check_mode:
            self._module.exit_json(changed=True)
        self._zapi.usermacro.createglobal({'macro': macro_name, 'value': macro_value, 'type': macro_type, 'description': macro_description})
        self._module.exit_json(changed=True, result='Successfully added global macro %s' % macro_name)
    except Exception as e:
        self._module.fail_json(msg='Failed to create global macro %s: %s' % (macro_name, e))