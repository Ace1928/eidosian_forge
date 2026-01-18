from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.community.zabbix.plugins.module_utils.base import ZabbixBase
from ansible.module_utils.compat.version import LooseVersion
import ansible_collections.community.zabbix.plugins.module_utils.helpers as zabbix_utils
def get_hostgroup_by_hostgroup_name(self, name):
    """Get host group by host group name.

        Parameters:
            name: Name of the host group.

        Returns:
            host group matching host group name.
        """
    try:
        _hostgroup = self._zapi.hostgroup.get({'output': 'extend', 'filter': {'name': [name]}})
        if len(_hostgroup) < 1:
            self._module.fail_json(msg='Host group not found: %s' % name)
        else:
            return _hostgroup[0]
    except Exception as e:
        self._module.fail_json(msg="Failed to get host group '%s': %s" % (name, e))