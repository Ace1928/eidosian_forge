from __future__ import absolute_import, division, print_function
import copy
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.community.network.plugins.module_utils.network.netscaler.netscaler import (
def gslb_service_identical(client, module, gslb_service_proxy):
    gslb_service_list = gslbservice.get_filtered(client, 'servicename:%s' % module.params['servicename'])
    diff_dict = gslb_service_proxy.diff_object(gslb_service_list[0])
    if 'ip' in diff_dict:
        del diff_dict['ip']
    if len(diff_dict) == 0:
        return True
    else:
        return False