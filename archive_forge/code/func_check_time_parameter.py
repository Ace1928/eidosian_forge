from __future__ import absolute_import, division, print_function
import re
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.community.zabbix.plugins.module_utils.base import ZabbixBase
import ansible_collections.community.zabbix.plugins.module_utils.helpers as zabbix_utils
def check_time_parameter(self, key_name, value):
    match_result = re.match('^[0-9]+[smhdw]$', value)
    if not match_result:
        self._module.fail_json(msg='Invalid value for %s! Please set value like 365d.' % key_name)