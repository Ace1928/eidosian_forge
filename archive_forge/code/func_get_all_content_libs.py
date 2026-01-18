from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.community.vmware.plugins.module_utils.vmware_rest_client import VmwareRestClient
def get_all_content_libs(self):
    """Method to retrieve List of content libraries."""
    content_libraries = self.local_content_libraries + self.subscribed_content_libraries
    self.module.exit_json(changed=False, content_libs=content_libraries)