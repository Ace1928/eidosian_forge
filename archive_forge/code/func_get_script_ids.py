from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.community.zabbix.plugins.module_utils.base import ZabbixBase
import ansible_collections.community.zabbix.plugins.module_utils.helpers as zabbix_utils
def get_script_ids(self, script_name):
    script_ids = []
    scripts = self._zapi.script.get({'filter': {'name': script_name}})
    for script in scripts:
        script_ids.append(script['scriptid'])
    return script_ids