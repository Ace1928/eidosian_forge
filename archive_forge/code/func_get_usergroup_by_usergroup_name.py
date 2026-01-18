from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.community.zabbix.plugins.module_utils.base import ZabbixBase
from ansible.module_utils.compat.version import LooseVersion
import ansible_collections.community.zabbix.plugins.module_utils.helpers as zabbix_utils
def get_usergroup_by_usergroup_name(self, name):
    """Get user group by user group name.

        Parameters:
            name: Name of the user group.

        Returns:
            User group matching user group name.
        """
    try:
        if LooseVersion(self._zbx_api_version) < LooseVersion('6.2'):
            _usergroup = self._zapi.usergroup.get({'output': 'extend', 'selectTagFilters': 'extend', 'selectRights': 'extend', 'filter': {'name': [name]}})
        else:
            _usergroup = self._zapi.usergroup.get({'output': 'extend', 'selectTagFilters': 'extend', 'selectHostGroupRights': 'extend', 'selectTemplateGroupRights': 'extend', 'filter': {'name': [name]}})
        if len(_usergroup) < 1:
            self._module.fail_json(msg='User group not found: %s' % name)
        else:
            return _usergroup[0]
    except Exception as e:
        self._module.fail_json(msg="Failed to get user group '%s': %s" % (name, e))