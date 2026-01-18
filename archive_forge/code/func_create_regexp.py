from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.community.zabbix.plugins.module_utils.base import ZabbixBase
from ansible.module_utils.compat.version import LooseVersion
import ansible_collections.community.zabbix.plugins.module_utils.helpers as zabbix_utils
def create_regexp(self, name, test_string, expressions):
    try:
        if self._module.check_mode:
            self._module.exit_json(changed=True)
        self._zapi.regexp.create({'name': name, 'test_string': test_string, 'expressions': self._convert_expressions_to_json(expressions)})
        self._module.exit_json(changed=True, msg='Successfully created regular expression setting.')
    except Exception as e:
        self._module.fail_json(msg='Failed to create regular expression setting: %s' % e)