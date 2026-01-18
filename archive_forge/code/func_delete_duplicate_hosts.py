from __future__ import (absolute_import, division, print_function)
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.community.zabbix.plugins.module_utils.base import ZabbixBase
import ansible_collections.community.zabbix.plugins.module_utils.helpers as zabbix_utils
def delete_duplicate_hosts(self, hosts):
    """ Delete duplicated hosts """
    unique_hosts = []
    listed_hostnames = []
    for zabbix_host in hosts:
        if zabbix_host['name'] in listed_hostnames:
            continue
        unique_hosts.append(zabbix_host)
        listed_hostnames.append(zabbix_host['name'])
    return unique_hosts