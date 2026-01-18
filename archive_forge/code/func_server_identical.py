from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.community.network.plugins.module_utils.network.netscaler.netscaler import (ConfigProxy, get_nitro_client,
def server_identical(client, module, server_proxy):
    log('Checking if configured server is identical')
    if server.count_filtered(client, 'name:%s' % module.params['name']) == 0:
        return False
    diff = diff_list(client, module, server_proxy)
    for option in ['graceful', 'delay']:
        if option in diff:
            del diff[option]
    if diff == {}:
        return True
    else:
        return False