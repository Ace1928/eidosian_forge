from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.community.zabbix.plugins.module_utils.base import ZabbixBase
from ansible.module_utils.compat.version import LooseVersion
import ansible_collections.community.zabbix.plugins.module_utils.helpers as zabbix_utils
def check_if_action_exists(self, name):
    """Check if action exists.

        Args:
            name: Name of the action.

        Returns:
            The return value. True for success, False otherwise.

        """
    try:
        _params = {'selectOperations': 'extend', 'selectRecoveryOperations': 'extend', 'selectUpdateOperations': 'extend', 'selectFilter': 'extend', 'filter': {'name': [name]}}
        _action = self._zapi.action.get(_params)
        return _action
    except Exception as e:
        self._module.fail_json(msg="Failed to check if action '%s' exists: %s" % (name, e))