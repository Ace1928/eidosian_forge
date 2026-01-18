from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.community.zabbix.plugins.module_utils.base import ZabbixBase
from ansible.module_utils.compat.version import LooseVersion
import ansible_collections.community.zabbix.plugins.module_utils.helpers as zabbix_utils
def delete_action(self, action_id):
    """Delete action.

        Args:
            action_id: Action id

        Returns:
            action: deleted action
        """
    try:
        if self._module.check_mode:
            self._module.exit_json(msg='Action would be deleted if check mode was not specified', changed=True)
        return self._zapi.action.delete([action_id])
    except Exception as e:
        self._module.fail_json(msg="Failed to delete action '%s': %s" % (action_id, e))