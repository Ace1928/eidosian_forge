from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.community.zabbix.plugins.module_utils.base import ZabbixBase
import ansible_collections.community.zabbix.plugins.module_utils.helpers as zabbix_utils
def check_if_valuemap_exists(self, name):
    """Checks if value map exists.

        Args:
            name: Zabbix valuemap name

        Returns:
            tuple: First element is True if valuemap exists and False otherwise.
                Second element is a dictionary of valuemap object if it exists.
        """
    try:
        valuemap_list = self._zapi.valuemap.get({'output': 'extend', 'selectMappings': 'extend', 'filter': {'name': [name]}})
        if len(valuemap_list) < 1:
            return (False, None)
        else:
            return (True, valuemap_list[0])
    except Exception as e:
        self._module.fail_json(msg="Failed to get ID of the valuemap '{name}': {e}".format(name=name, e=e))