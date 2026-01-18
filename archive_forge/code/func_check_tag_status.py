from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.community.vmware.plugins.module_utils.vmware_rest_client import VmwareRestClient
def check_tag_status(self):
    """
        Check if tag exists or not
        Returns: 'present' if tag found, else 'absent'

        """
    return 'present' if self.tag_obj is not None else 'absent'