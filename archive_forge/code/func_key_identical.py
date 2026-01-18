from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.community.network.plugins.module_utils.network.netscaler.netscaler import (ConfigProxy, get_nitro_client, netscaler_common_arguments,
def key_identical(client, module, sslcertkey_proxy):
    log('Checking if configured key is identical')
    sslcertkey_list = sslcertkey.get_filtered(client, 'certkey:%s' % module.params['certkey'])
    diff_dict = sslcertkey_proxy.diff_object(sslcertkey_list[0])
    if 'password' in diff_dict:
        del diff_dict['password']
    if 'passplain' in diff_dict:
        del diff_dict['passplain']
    if len(diff_dict) == 0:
        return True
    else:
        return False