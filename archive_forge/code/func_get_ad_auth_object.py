from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.community.vmware.plugins.module_utils.vmware import PyVmomi, TaskError, vmware_argument_spec, wait_for_task
from ansible.module_utils._text import to_native
def get_ad_auth_object(self, host_object):
    """Get AD authentication managed object"""
    ad_authentication = None
    authentication_store_info = host_object.configManager.authenticationManager.supportedStore
    for store_info in authentication_store_info:
        if isinstance(store_info, vim.host.ActiveDirectoryAuthentication):
            ad_authentication = store_info
            break
    if not ad_authentication:
        self.module.fail_json(msg='Failed to get Active Directory authentication managed object from authentication manager')
    return ad_authentication