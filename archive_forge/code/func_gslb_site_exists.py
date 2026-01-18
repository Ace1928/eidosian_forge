from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.community.network.plugins.module_utils.network.netscaler.netscaler import (
def gslb_site_exists(client, module):
    if gslbsite.count_filtered(client, 'sitename:%s' % module.params['sitename']) > 0:
        return True
    else:
        return False