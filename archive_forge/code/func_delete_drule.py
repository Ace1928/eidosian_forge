from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.community.zabbix.plugins.module_utils.base import ZabbixBase
from ansible.module_utils.compat.version import LooseVersion
import ansible_collections.community.zabbix.plugins.module_utils.helpers as zabbix_utils
def delete_drule(self, drule_id):
    """Delete discovery rule.
        Args:
            drule_id: Discovery rule id
        Returns:
            drule: deleted discovery rule
        """
    try:
        if self._module.check_mode:
            self._module.exit_json(changed=True, msg='Discovery rule would be deleted if check mode was not specified')
        return self._zapi.drule.delete([drule_id])
    except Exception as e:
        self._module.fail_json(msg="Failed to delete discovery rule '%s': %s" % (drule_id, e))