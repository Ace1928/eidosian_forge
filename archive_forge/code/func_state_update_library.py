from __future__ import absolute_import, division, print_function
import uuid
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.community.vmware.plugins.module_utils.vmware_rest_client import VmwareRestClient
from ansible_collections.community.vmware.plugins.module_utils.vmware import PyVmomi
def state_update_library(self):
    """
        Update Content Library

        """
    self.fail_when_duplicated()
    changed = False
    library_id = self.local_libraries[self.library_name]['lib_id']
    library_update_spec = LibraryModel()
    existing_library_type = self.local_libraries[self.library_name]['lib_type'].lower()
    if existing_library_type != self.library_type:
        self.module.fail_json(msg='Library [%s] is of type %s, cannot be changed to %s' % (self.library_name, existing_library_type, self.library_type))
    if self.library_type == 'subscribed':
        existing_subscription_url = self.local_libraries[self.library_name]['lib_sub_url']
        sub_url_changed = existing_subscription_url != self.subscription_url
        existing_on_demand = self.local_libraries[self.library_name]['lib_sub_on_demand']
        sub_on_demand_changed = existing_on_demand != self.update_on_demand
        sub_ssl_thumbprint_changed = False
        if 'https:' in self.subscription_url and self.ssl_thumbprint:
            existing_ssl_thumbprint = self.local_libraries[self.library_name]['lib_sub_ssl_thumbprint']
            sub_ssl_thumbprint_changed = existing_ssl_thumbprint != self.ssl_thumbprint
        if sub_url_changed or sub_on_demand_changed or sub_ssl_thumbprint_changed:
            subscription_info = self.set_subscription_spec()
            library_update_spec.subscription_info = subscription_info
            changed = True
    library_desc = self.local_libraries[self.library_name]['lib_description']
    desired_lib_desc = self.params.get('library_description')
    if library_desc != desired_lib_desc:
        library_update_spec.description = desired_lib_desc
        changed = True
    if changed:
        library_update_spec.name = self.library_name
        self.create_update(spec=library_update_spec, library_id=library_id, update=True)
    content_library_info = dict(msg='Content Library %s is unchanged.' % self.library_name, library_id=library_id)
    self.module.exit_json(changed=False, content_library_info=dict(msg=content_library_info, library_id=library_id))