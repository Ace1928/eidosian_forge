import os
import testinfra.utils.ansible_runner
def test_docker_running(host):
    zabbixagent = host.docker('zabbix-agent')
    zabbixagent.is_running