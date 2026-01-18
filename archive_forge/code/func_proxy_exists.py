from __future__ import (absolute_import, division, print_function)
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.community.zabbix.plugins.module_utils.base import ZabbixBase
import ansible_collections.community.zabbix.plugins.module_utils.helpers as zabbix_utils
def proxy_exists(self, proxy_name):
    result = self._zapi.proxy.get({'output': 'extend', 'selectInterface': 'extend', 'filter': {'host': proxy_name}})
    if len(result) > 0 and 'proxyid' in result[0]:
        self.existing_data = result[0]
        return result[0]['proxyid']
    else:
        return result