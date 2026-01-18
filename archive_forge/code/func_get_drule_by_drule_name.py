from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.community.zabbix.plugins.module_utils.base import ZabbixBase
from ansible.module_utils.compat.version import LooseVersion
import ansible_collections.community.zabbix.plugins.module_utils.helpers as zabbix_utils
def get_drule_by_drule_name(self, name):
    """Get discovery rule by discovery rule name
        Args:
            name: discovery rule name.
        Returns:
            discovery rule matching discovery rule name
        """
    try:
        drule_list = self._zapi.drule.get({'output': 'extend', 'selectDChecks': 'extend', 'filter': {'name': [name]}})
        if len(drule_list) < 1:
            self._module.fail_json(msg='Discovery rule not found: %s' % name)
        else:
            return drule_list[0]
    except Exception as e:
        self._module.fail_json(msg="Failed to get discovery rule '%s': %s" % (name, e))