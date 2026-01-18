from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.purestorage.flashblade.plugins.module_utils.purefb import (
from datetime import datetime
def generate_ad_dict(blade):
    ad_info = {}
    active_directory = blade.get_active_directory()
    if active_directory.total_item_count != 0:
        ad_account = list(active_directory.items)[0]
        ad_info[ad_account.name] = {'computer': ad_account.computer_name, 'domain': ad_account.domain, 'directory_servers': ad_account.directory_servers, 'kerberos_servers': ad_account.kerberos_servers, 'service_principals': ad_account.service_principal_names, 'join_ou': ad_account.join_ou, 'encryption_types': ad_account.encryption_types, 'global_catalog_servers': getattr(ad_account, 'global_catalog_servers', None)}
    return ad_info