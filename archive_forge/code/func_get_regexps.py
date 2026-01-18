from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.community.zabbix.plugins.module_utils.base import ZabbixBase
from ansible.module_utils.compat.version import LooseVersion
import ansible_collections.community.zabbix.plugins.module_utils.helpers as zabbix_utils
def get_regexps(self, regexp_name):
    try:
        regexps = self._zapi.regexp.get({'output': 'extend', 'selectExpressions': ['expression', 'expression_type', 'exp_delimiter', 'case_sensitive'], 'filter': {'name': regexp_name}})
        if len(regexps) >= 2:
            self._module.fail_json('Too many regexps are matched.')
        return regexps
    except Exception as e:
        self._module.fail_json(msg='Failed to get regular expression setting: %s' % e)