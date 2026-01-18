from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.community.zabbix.plugins.module_utils.base import ZabbixBase
import ansible_collections.community.zabbix.plugins.module_utils.helpers as zabbix_utils
def get_service_ids(self, service_name):
    service_ids = []
    services = self._zapi.service.get({'filter': {'name': service_name}})
    for service in services:
        service_ids.append(service['serviceid'])
    return service_ids