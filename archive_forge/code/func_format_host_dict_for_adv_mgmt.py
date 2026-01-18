from __future__ import absolute_import, division, print_function
import re
import traceback
from ansible.module_utils.basic import AnsibleModule, missing_required_lib
from ansible_collections.dellemc.unity.plugins.module_utils.storage.dell \
def format_host_dict_for_adv_mgmt(self):
    """
        Form host access for advance management
        :return: Formatted Host access type info
        :rtype: dict
        """
    result_host = {}
    for param in list(self.host_param_mapping.keys()):
        if self.module.params[param]:
            result_host[param] = []
            for host_dict in self.module.params[param]:
                result_host[param].append(self.get_host_obj_value(host_dict))
    if 'read_only_root_hosts' in result_host:
        result_host['read_only_root_access_hosts'] = result_host.pop('read_only_root_hosts')
    if 'read_write_root_hosts' in result_host:
        result_host['root_access_hosts'] = result_host.pop('read_write_root_hosts')
    return result_host