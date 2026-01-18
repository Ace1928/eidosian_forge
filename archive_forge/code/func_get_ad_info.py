from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.community.vmware.plugins.module_utils.vmware import PyVmomi, TaskError, vmware_argument_spec, wait_for_task
from ansible.module_utils._text import to_native
def get_ad_info(self, host_object):
    """Get info about AD membership"""
    active_directory_info = None
    authentication_store_info = host_object.config.authenticationManagerInfo.authConfig
    for authentication_info in authentication_store_info:
        if isinstance(authentication_info, vim.host.ActiveDirectoryInfo):
            active_directory_info = authentication_info
            break
    if not active_directory_info:
        self.module.fail_json(msg='Failed to get Active Directory info from authentication manager')
    return active_directory_info