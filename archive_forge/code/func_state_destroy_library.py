from __future__ import absolute_import, division, print_function
import uuid
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.community.vmware.plugins.module_utils.vmware_rest_client import VmwareRestClient
from ansible_collections.community.vmware.plugins.module_utils.vmware import PyVmomi
def state_destroy_library(self):
    """
        Delete Content Library

        """
    self.fail_when_duplicated()
    library_id = self.local_libraries[self.library_name]['lib_id']
    library_service = self.library_types[self.local_libraries[self.library_name]['lib_type'].lower()]
    if self.module.check_mode:
        action = 'would be deleted'
    else:
        action = 'deleted'
        library_service.delete(library_id=library_id)
    self.module.exit_json(changed=True, content_library_info=dict(msg="Content Library '%s' %s." % (self.library_name, action), library_id=library_id))