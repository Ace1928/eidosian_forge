from __future__ import (absolute_import, division, print_function)
from ansible.module_utils.basic import AnsibleModule, missing_required_lib
from ansible_collections.infinidat.infinibox.plugins.module_utils.infinibox import (
def disable_and_delete_ip(module, network_space, ip):
    """
    Disable and delete a network space IP
    """
    if not ip:
        return
    addr = ip['ip_address']
    network_space_name = module.params['name']
    ip_type = ip['type']
    mgmt = ''
    if ip_type == 'MANAGEMENT':
        mgmt = 'management '
    try:
        try:
            network_space.disable_ip_address(addr)
        except APICommandFailed as err:
            if err.error_code == 'IP_ADDRESS_ALREADY_DISABLED':
                pass
            else:
                module.fail_json(msg=f'Disabling of network space {network_space_name} IP {mgmt}{addr} API command failed')
        network_space.remove_ip_address(addr)
    except Exception as err:
        module.fail_json(msg=f'Disabling or removal of network space {network_space_name} IP {mgmt}{addr} failed: {err}')