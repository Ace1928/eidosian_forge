from __future__ import absolute_import, division, print_function
import uuid
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.community.vmware.plugins.module_utils.vmware_rest_client import VmwareRestClient
from ansible_collections.community.vmware.plugins.module_utils.vmware import PyVmomi
def fail_when_duplicated(self):
    if self.existing_library_names.count(self.library_name) > 1:
        self.module.fail_json(msg='Operation cannot continue, library [%s] is not unique' % self.library_name)