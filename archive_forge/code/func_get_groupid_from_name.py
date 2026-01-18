from __future__ import absolute_import, division, print_function
import re
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.community.zabbix.plugins.module_utils.base import ZabbixBase
from ansible.module_utils.compat.version import LooseVersion
import ansible_collections.community.zabbix.plugins.module_utils.helpers as zabbix_utils
def get_groupid_from_name(self, hostgroup):
    groupid = self._zapi.hostgroup.get({'filter': {'name': hostgroup}})
    if not groupid or len(groupid) > 1:
        self._module.fail_json("Host group '%s' cannot be found" % hostgroup)
    return groupid[0]['groupid']