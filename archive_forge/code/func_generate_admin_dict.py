from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.purestorage.flashblade.plugins.module_utils.purefb import (
from datetime import datetime
def generate_admin_dict(module, blade):
    admin_info = {}
    api_version = blade.api_version.list_versions().versions
    if MULTIPROTOCOL_API_VERSION in api_version:
        admins = blade.admins.list_admins()
        for admin in range(0, len(admins.items)):
            admin_name = admins.items[admin].name
            admin_info[admin_name] = {'api_token_timeout': admins.items[admin].api_token_timeout, 'public_key': admins.items[admin].public_key, 'local': admins.items[admin].is_local}
    if MIN_32_API in api_version:
        bladev2 = get_system(module)
        admins = list(bladev2.get_admins().items)
        for admin in range(0, len(admins)):
            admin_name = admins[admin].name
            if admins[admin].api_token.expires_at:
                admin_info[admin_name]['token_expires'] = datetime.fromtimestamp(admins[admin].api_token.expires_at / 1000).strftime('%Y-%m-%d %H:%M:%S')
            else:
                admin_info[admin_name]['token_expires'] = None
            admin_info[admin_name]['token_created'] = datetime.fromtimestamp(admins[admin].api_token.created_at / 1000).strftime('%Y-%m-%d %H:%M:%S')
            admin_info[admin_name]['role'] = admins[admin].role.name
            if NFS_POLICY_API_VERSION in api_version:
                admin_info[admin_name]['locked'] = admins[admin].locked
                admin_info[admin_name]['lockout_remaining'] = admins[admin].lockout_remaining
    return admin_info