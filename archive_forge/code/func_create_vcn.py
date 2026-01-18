from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule, missing_required_lib
from ansible_collections.community.general.plugins.module_utils.oracle import oci_utils
def create_vcn(virtual_network_client, module):
    create_vcn_details = CreateVcnDetails()
    for attribute in create_vcn_details.attribute_map.keys():
        if attribute in module.params:
            setattr(create_vcn_details, attribute, module.params[attribute])
    result = oci_utils.create_and_wait(resource_type='vcn', create_fn=virtual_network_client.create_vcn, kwargs_create={'create_vcn_details': create_vcn_details}, client=virtual_network_client, get_fn=virtual_network_client.get_vcn, get_param='vcn_id', module=module)
    return result