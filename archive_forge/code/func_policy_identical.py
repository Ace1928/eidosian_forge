from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.community.network.plugins.module_utils.network.netscaler.netscaler import (ConfigProxy, get_nitro_client, netscaler_common_arguments,
def policy_identical(client, module, cspolicy_proxy):
    log('Checking if defined policy is identical to configured')
    if cspolicy.count_filtered(client, 'policyname:%s' % module.params['policyname']) == 0:
        return False
    policy_list = cspolicy.get_filtered(client, 'policyname:%s' % module.params['policyname'])
    diff_dict = cspolicy_proxy.diff_object(policy_list[0])
    if 'ip' in diff_dict:
        del diff_dict['ip']
    if len(diff_dict) == 0:
        return True
    else:
        return False