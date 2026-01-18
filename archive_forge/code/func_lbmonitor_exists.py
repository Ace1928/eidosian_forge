from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.community.network.plugins.module_utils.network.netscaler.netscaler import (
def lbmonitor_exists(client, module):
    log('Checking if monitor exists')
    if lbmonitor.count_filtered(client, 'monitorname:%s' % module.params['monitorname']) > 0:
        return True
    else:
        return False