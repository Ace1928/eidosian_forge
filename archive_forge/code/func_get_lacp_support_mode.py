from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils._text import to_native
from ansible_collections.community.vmware.plugins.module_utils.vmware import (
@staticmethod
def get_lacp_support_mode(mode):
    """Get LACP support mode"""
    return_mode = None
    if mode == 'basic':
        return_mode = 'singleLag'
    elif mode == 'enhanced':
        return_mode = 'multipleLag'
    elif mode == 'singleLag':
        return_mode = 'basic'
    elif mode == 'multipleLag':
        return_mode = 'enhanced'
    return return_mode