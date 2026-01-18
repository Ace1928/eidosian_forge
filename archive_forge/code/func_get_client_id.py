from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.purestorage.fusion.plugins.module_utils.fusion import (
from ansible_collections.purestorage.fusion.plugins.module_utils.startup import (
def get_client_id(module, fusion):
    """Get API Client ID, or None if not available"""
    id_api_instance = purefusion.IdentityManagerApi(fusion)
    try:
        clients = id_api_instance.list_api_clients()
        for client in clients:
            if client.public_key == module.params['public_key'] and client.display_name == module.params['name']:
                return client.id
        return None
    except purefusion.rest.ApiException:
        return None