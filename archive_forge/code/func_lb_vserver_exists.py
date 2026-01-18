from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.community.network.plugins.module_utils.network.netscaler.netscaler import (
import copy
def lb_vserver_exists(client, module):
    log('Checking if lb vserver exists')
    if lbvserver.count_filtered(client, 'name:%s' % module.params['name']) > 0:
        return True
    else:
        return False