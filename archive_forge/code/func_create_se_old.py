from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.purestorage.fusion.plugins.module_utils.fusion import (
from ansible_collections.purestorage.fusion.plugins.module_utils.networking import (
from ansible_collections.purestorage.fusion.plugins.module_utils.startup import (
from ansible_collections.purestorage.fusion.plugins.module_utils.operations import (
def create_se_old(module, fusion):
    """Create Storage Endpoint"""
    se_api_instance = purefusion.StorageEndpointsApi(fusion)
    changed = True
    if not module.check_mode:
        if not module.params['display_name']:
            display_name = module.params['name']
        else:
            display_name = module.params['display_name']
        ifaces = []
        for address in module.params['addresses']:
            if module.params['gateway']:
                iface = purefusion.StorageEndpointIscsiDiscoveryInterfacePost(address=address, gateway=module.params['gateway'], network_interface_groups=module.params['network_interface_groups'])
            else:
                iface = purefusion.StorageEndpointIscsiDiscoveryInterfacePost(address=address, network_interface_groups=module.params['network_interface_groups'])
            ifaces.append(iface)
        op = purefusion.StorageEndpointPost(endpoint_type='iscsi', iscsi=purefusion.StorageEndpointIscsiPost(discovery_interfaces=ifaces), name=module.params['name'], display_name=display_name)
        op = se_api_instance.create_storage_endpoint(op, region_name=module.params['region'], availability_zone_name=module.params['availability_zone'])
        await_operation(fusion, op)
    module.exit_json(changed=changed)