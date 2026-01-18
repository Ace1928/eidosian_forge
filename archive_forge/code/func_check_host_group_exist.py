from __future__ import absolute_import, division, print_function
import copy
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.community.zabbix.plugins.module_utils.base import ZabbixBase
import ansible_collections.community.zabbix.plugins.module_utils.helpers as zabbix_utils
def check_host_group_exist(self, group_names):
    for group_name in group_names:
        result = self._zapi.hostgroup.get({'filter': {'name': group_name}})
        if not result:
            self._module.fail_json(msg='Hostgroup not found: %s' % group_name)
    return True