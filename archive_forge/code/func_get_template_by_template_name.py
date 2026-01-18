from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.community.zabbix.plugins.module_utils.base import ZabbixBase
from ansible.module_utils.compat.version import LooseVersion
import ansible_collections.community.zabbix.plugins.module_utils.helpers as zabbix_utils
def get_template_by_template_name(self, template_name):
    """Get template by template name

        Args:
            template_name: template name.

        Returns:
            template matching template name

        """
    try:
        template_list = self._zapi.template.get({'output': 'extend', 'filter': {'host': [template_name]}})
        if len(template_list) < 1:
            self._module.fail_json(msg='Template not found: %s' % template_name)
        else:
            return template_list[0]
    except Exception as e:
        self._module.fail_json(msg="Failed to get template '%s': %s" % (template_name, e))