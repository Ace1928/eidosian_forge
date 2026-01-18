from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.purestorage.flashblade.plugins.module_utils.purefb import (
from datetime import datetime
def generate_remote_creds_dict(blade):
    remote_creds_info = {}
    remote_creds = blade.object_store_remote_credentials.list_object_store_remote_credentials()
    for cred_cnt in range(0, len(remote_creds.items)):
        cred_name = remote_creds.items[cred_cnt].name
        remote_creds_info[cred_name] = {'access_key': remote_creds.items[cred_cnt].access_key_id, 'remote_array': remote_creds.items[cred_cnt].remote.name}
    return remote_creds_info