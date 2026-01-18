from __future__ import absolute_import, division, print_function
import uuid
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.community.vmware.plugins.module_utils.vmware_rest_client import VmwareRestClient
from ansible_collections.community.vmware.plugins.module_utils.vmware import PyVmomi
def get_all_libraries(self, library_service):
    content_libs = library_service.list()
    if content_libs:
        for content_lib in content_libs:
            lib_details = library_service.get(content_lib)
            lib_dict = dict(lib_name=lib_details.name, lib_description=lib_details.description, lib_id=lib_details.id, lib_type=lib_details.type)
            if lib_details.type == 'SUBSCRIBED':
                lib_dict['lib_sub_url'] = lib_details.subscription_info.subscription_url
                lib_dict['lib_sub_on_demand'] = lib_details.subscription_info.on_demand
                lib_dict['lib_sub_ssl_thumbprint'] = lib_details.subscription_info.ssl_thumbprint
            self.local_libraries[lib_details.name] = lib_dict
            self.existing_library_names.append(lib_details.name)