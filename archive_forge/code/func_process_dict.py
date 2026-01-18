from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.dellemc.unity.plugins.module_utils.storage.dell \
def process_dict(nfs_server_details):
    """Process NFS server details.
        :param: nfs_server_details: Dict containing NFS server details
        :return: Processed dict containing NFS server details
    """
    param_list = ['credentials_cache_ttl', 'file_interfaces', 'host_name', 'id', 'kdc_type', 'nas_server', 'is_secure_enabled', 'is_extended_credentials_enabled', 'nfs_v4_enabled', 'servicee_principal_name']
    for param in param_list:
        if param in nfs_server_details and param == 'credentials_cache_ttl':
            nfs_server_details[param] = str(nfs_server_details[param][0])
        else:
            nfs_server_details[param] = nfs_server_details[param][0]
    return nfs_server_details