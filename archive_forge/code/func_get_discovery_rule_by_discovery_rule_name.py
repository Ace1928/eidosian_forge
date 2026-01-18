from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.community.zabbix.plugins.module_utils.base import ZabbixBase
from ansible.module_utils.compat.version import LooseVersion
import ansible_collections.community.zabbix.plugins.module_utils.helpers as zabbix_utils
def get_discovery_rule_by_discovery_rule_name(self, discovery_rule_name):
    """Get discovery rule by discovery rule name

        Args:
            discovery_rule_name: discovery rule name.

        Returns:
            discovery rule matching discovery rule name

        """
    try:
        discovery_rule_list = self._zapi.drule.get({'output': 'extend', 'filter': {'name': [discovery_rule_name]}})
        if len(discovery_rule_list) < 1:
            self._module.fail_json(msg='Discovery rule not found: %s' % discovery_rule_name)
        else:
            return discovery_rule_list[0]
    except Exception as e:
        self._module.fail_json(msg="Failed to get discovery rule '%s': %s" % (discovery_rule_name, e))