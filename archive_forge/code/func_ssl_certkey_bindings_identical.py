from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.community.network.plugins.module_utils.network.netscaler.netscaler import (
def ssl_certkey_bindings_identical(client, module):
    log('Checking if ssl cert key bindings are identical')
    vservername = module.params['name']
    if sslvserver_sslcertkey_binding.count(client, vservername) == 0:
        bindings = []
    else:
        bindings = sslvserver_sslcertkey_binding.get(client, vservername)
    if module.params['ssl_certkey'] is None:
        if len(bindings) == 0:
            return True
        else:
            return False
    else:
        certificate_list = [item.certkeyname for item in bindings]
        if certificate_list == [module.params['ssl_certkey']]:
            return True
        else:
            return False