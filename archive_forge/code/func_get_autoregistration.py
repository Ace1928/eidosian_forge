from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.community.zabbix.plugins.module_utils.base import ZabbixBase
import ansible_collections.community.zabbix.plugins.module_utils.helpers as zabbix_utils
def get_autoregistration(self):
    try:
        return self._zapi.autoregistration.get({'output': 'extend'})
    except Exception as e:
        self._module.fail_json(msg='Failed to get autoregistration: %s' % e)