from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.community.zabbix.plugins.module_utils.base import ZabbixBase
from ansible.module_utils.compat.version import LooseVersion
import ansible_collections.community.zabbix.plugins.module_utils.helpers as zabbix_utils
def delete_regexp(self, regexp):
    try:
        parameter = [regexp['regexpid']]
        if self._module.check_mode:
            self._module.exit_json(changed=True)
        self._zapi.regexp.delete(parameter)
        self._module.exit_json(changed=True, msg='Successfully deleted regular expression setting.')
    except Exception as e:
        self._module.fail_json(msg='Failed to delete regular expression setting: %s' % e)