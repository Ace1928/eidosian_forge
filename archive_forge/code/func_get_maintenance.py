from __future__ import absolute_import, division, print_function
import datetime
import time
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.community.zabbix.plugins.module_utils.base import ZabbixBase
import ansible_collections.community.zabbix.plugins.module_utils.helpers as zabbix_utils
def get_maintenance(self, name):
    maintenances = self._zapi.maintenance.get({'filter': {'name': name}, 'selectGroups': 'extend', 'selectHosts': 'extend', 'selectTags': 'extend'})
    for maintenance in maintenances:
        maintenance['groupids'] = [group['groupid'] for group in maintenance['groups']] if 'groups' in maintenance else []
        maintenance['hostids'] = [host['hostid'] for host in maintenance['hosts']] if 'hosts' in maintenance else []
        return (0, maintenance, None)
    return (0, None, None)