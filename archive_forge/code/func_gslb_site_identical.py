from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.community.network.plugins.module_utils.network.netscaler.netscaler import (
def gslb_site_identical(client, module, gslb_site_proxy):
    gslb_site_list = gslbsite.get_filtered(client, 'sitename:%s' % module.params['sitename'])
    diff_dict = gslb_site_proxy.diff_object(gslb_site_list[0])
    if len(diff_dict) == 0:
        return True
    else:
        return False