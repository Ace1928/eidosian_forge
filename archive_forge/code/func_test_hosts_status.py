import os
from zabbix_api import ZabbixAPI
import testinfra.utils.ansible_runner
def test_hosts_status():
    zapi = authenticate()
    servers = zapi.host.get({'output': ['status', 'name']})
    for server in servers:
        if server['name'] != 'Zabbix server':
            assert int(server['status']) == 0