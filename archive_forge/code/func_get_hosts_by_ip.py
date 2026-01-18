from __future__ import (absolute_import, division, print_function)
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.community.zabbix.plugins.module_utils.base import ZabbixBase
import ansible_collections.community.zabbix.plugins.module_utils.helpers as zabbix_utils
def get_hosts_by_ip(self, host_ips, host_inventory):
    """ Get host by host ip(s) """
    hostinterfaces = self._zapi.hostinterface.get({'output': 'extend', 'filter': {'ip': host_ips}})
    if len(hostinterfaces) < 1:
        self._module.fail_json(msg='Host not found: %s' % host_ips)
    host_list = []
    for hostinterface in hostinterfaces:
        host = self._zapi.host.get({'output': 'extend', 'selectGroups': 'extend', 'selectParentTemplates': ['name'], 'hostids': hostinterface['hostid'], 'selectInventory': host_inventory, 'selectTags': 'extend', 'selectMacros': 'extend'})
        host[0]['hostinterfaces'] = hostinterface
        host_list.append(host[0])
    return host_list