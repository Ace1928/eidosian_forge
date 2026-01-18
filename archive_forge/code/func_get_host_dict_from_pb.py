from __future__ import absolute_import, division, print_function
import re
import traceback
from ansible.module_utils.basic import AnsibleModule, missing_required_lib
from ansible_collections.dellemc.unity.plugins.module_utils.storage.dell \
def get_host_dict_from_pb(self):
    """ Traverse all given hosts params and provides with host dict,
            which has respective host str param name with its value
            required by SDK

        :return: dict with key named as respective host str param name & value
                required by SDK
        :rtype: dict
        """
    LOG.info('Getting host parameters')
    result_host = {}
    if self.module.params['host_state']:
        if not self.module.params['adv_host_mgmt_enabled']:
            result_host = self.format_host_dict_for_non_adv_mgmt()
        else:
            result_host = self.format_host_dict_for_adv_mgmt()
    return result_host