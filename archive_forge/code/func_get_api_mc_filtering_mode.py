from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils._text import to_native
from ansible_collections.community.vmware.plugins.module_utils.vmware import (
@staticmethod
def get_api_mc_filtering_mode(mode):
    """Get Multicast filtering mode"""
    if mode == 'basic':
        return 'legacyFiltering'
    return 'snooping'