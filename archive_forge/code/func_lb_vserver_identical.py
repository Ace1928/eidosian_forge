from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.community.network.plugins.module_utils.network.netscaler.netscaler import (
import copy
def lb_vserver_identical(client, module, lbvserver_proxy):
    log('Checking if configured lb vserver is identical')
    lbvserver_list = lbvserver.get_filtered(client, 'name:%s' % module.params['name'])
    if lbvserver_proxy.has_equal_attributes(lbvserver_list[0]):
        return True
    else:
        return False