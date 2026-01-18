from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule, missing_required_lib
from ansible_collections.community.general.plugins.module_utils.oracle import oci_utils
def delete_vcn(virtual_network_client, module):
    result = oci_utils.delete_and_wait(resource_type='vcn', client=virtual_network_client, get_fn=virtual_network_client.get_vcn, kwargs_get={'vcn_id': module.params['vcn_id']}, delete_fn=virtual_network_client.delete_vcn, kwargs_delete={'vcn_id': module.params['vcn_id']}, module=module)
    return result