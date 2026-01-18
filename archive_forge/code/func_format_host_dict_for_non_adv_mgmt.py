from __future__ import absolute_import, division, print_function
import re
import traceback
from ansible.module_utils.basic import AnsibleModule, missing_required_lib
from ansible_collections.dellemc.unity.plugins.module_utils.storage.dell \
def format_host_dict_for_non_adv_mgmt(self):
    """
        Form host access for non advance management option
        :return: Formatted Host access type info
        :rtype: dict
        """
    result_host = {}
    for param in list(self.host_param_mapping.keys()):
        if self.module.params[param]:
            result_host[param] = ''
            for host_dict in self.module.params[param]:
                result_host[param] += self.get_host_access_string_value(host_dict)
    if result_host != {}:
        result_host = {self.host_param_mapping[k]: v[:-1] for k, v in result_host.items()}
    return result_host