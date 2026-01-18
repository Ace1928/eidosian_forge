from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.community.zabbix.plugins.module_utils.base import ZabbixBase
from ansible.module_utils.compat.version import LooseVersion
import ansible_collections.community.zabbix.plugins.module_utils.helpers as zabbix_utils
def check_difference(self, **kwargs):
    """Check difference between user group and user specified parameters.

        Parameters:
            **kwargs: Arbitrary keyword parameters.

        Returns:
            dict: dictionary of differences
        """
    existing_usergroup = zabbix_utils.helper_convert_unicode_to_str(self.get_usergroup_by_usergroup_name(kwargs['name']))
    parameters = zabbix_utils.helper_convert_unicode_to_str(self._construct_parameters(**kwargs))
    change_parameters = {}
    _diff = zabbix_utils.helper_compare_dictionaries(parameters, existing_usergroup, change_parameters)
    return _diff