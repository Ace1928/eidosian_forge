from __future__ import absolute_import, division, print_function
import copy
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.community.zabbix.plugins.module_utils.base import ZabbixBase
import ansible_collections.community.zabbix.plugins.module_utils.helpers as zabbix_utils
def get_group_ids_by_host_id(self, host_id):
    return self._zapi.hostgroup.get({'output': 'extend', 'hostids': host_id})