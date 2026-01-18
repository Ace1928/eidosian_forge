from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.community.zabbix.plugins.module_utils.base import ZabbixBase
from ansible.module_utils.compat.version import LooseVersion
import ansible_collections.community.zabbix.plugins.module_utils.helpers as zabbix_utils
def construct_the_data(self, _tag_filters):
    """Construct the user defined tag filters to fit the Zabbix API requirements

        Parameters:
            _tag_filters: tag filters to construct

        Returns:
            dict: user defined tag filters
        """
    if _tag_filters is None:
        return []
    constructed_data = []
    for tag_filter in _tag_filters:
        constructed_tag_filter = {'groupid': self.get_hostgroup_by_hostgroup_name(tag_filter.get('host_group'))['groupid'], 'tag': tag_filter.get('tag'), 'value': tag_filter.get('value')}
        constructed_data.append(constructed_tag_filter)
    return zabbix_utils.helper_cleanup_data(constructed_data)