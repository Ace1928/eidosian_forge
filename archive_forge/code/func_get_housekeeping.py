from __future__ import absolute_import, division, print_function
import re
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.community.zabbix.plugins.module_utils.base import ZabbixBase
import ansible_collections.community.zabbix.plugins.module_utils.helpers as zabbix_utils
def get_housekeeping(self):
    try:
        return self._zapi.housekeeping.get({'output': 'extend'})
    except Exception as e:
        self._module.fail_json(msg='Failed to get housekeeping setting: %s' % e)