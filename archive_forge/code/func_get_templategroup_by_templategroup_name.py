from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.community.zabbix.plugins.module_utils.base import ZabbixBase
from ansible.module_utils.compat.version import LooseVersion
import ansible_collections.community.zabbix.plugins.module_utils.helpers as zabbix_utils
def get_templategroup_by_templategroup_name(self, name):
    """Get template group by template group name.

        Parameters:
            name: Name of the template group.

        Returns:
            template group matching template group name.
        """
    try:
        _templategroup = self._zapi.templategroup.get({'output': 'extend', 'filter': {'name': [name]}})
        if len(_templategroup) < 1:
            self._module.fail_json(msg='Template group not found: %s' % name)
        else:
            return _templategroup[0]
    except Exception as e:
        self._module.fail_json(msg="Failed to get template group '%s': %s" % (name, e))