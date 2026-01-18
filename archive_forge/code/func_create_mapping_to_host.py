from __future__ import (absolute_import, division, print_function)
from ansible.module_utils.basic import AnsibleModule, missing_required_lib
from ansible_collections.infinidat.infinibox.plugins.module_utils.infinibox import (
@api_wrapper
def create_mapping_to_host(module, system):
    """ Create mapping of volume to host. If already mapped, exit_json with changed False. """
    changed = False
    host = system.hosts.get(name=module.params['host'])
    volume = get_volume(module, system)
    volume_name = module.params['volume']
    host_name = module.params['host']
    lun_name = module.params['lun']
    lun_use = find_host_lun_use(module, host, volume)
    if lun_use['lun_used']:
        msg = f"Cannot create mapping of volume '{volume_name}' to host '{host_name}' using lun '{lun_name}'. Lun in use."
        module.fail_json(msg=msg)
    try:
        desired_lun = module.params['lun']
        if not module.check_mode:
            host.map_volume(volume, lun=desired_lun)
        changed = True
    except APICommandFailed as err:
        if 'is already mapped' not in str(err):
            msg = f"Cannot map volume '{host_name}' to host '{host_name}': {str(err)}. Already mapped."
            module.fail_json(msg=msg)
    return changed