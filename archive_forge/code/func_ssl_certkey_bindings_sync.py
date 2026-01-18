from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.community.network.plugins.module_utils.network.netscaler.netscaler import (
def ssl_certkey_bindings_sync(client, module):
    log('Syncing certkey bindings')
    vservername = module.params['name']
    if sslvserver_sslcertkey_binding.count(client, vservername) == 0:
        bindings = []
    else:
        bindings = sslvserver_sslcertkey_binding.get(client, vservername)
    for binding in bindings:
        log('Deleting existing binding for certkey %s' % binding.certkeyname)
        sslvserver_sslcertkey_binding.delete(client, binding)
    if module.params['ssl_certkey'] is not None:
        log('Adding binding for certkey %s' % module.params['ssl_certkey'])
        binding = sslvserver_sslcertkey_binding()
        binding.vservername = module.params['name']
        binding.certkeyname = module.params['ssl_certkey']
        sslvserver_sslcertkey_binding.add(client, binding)