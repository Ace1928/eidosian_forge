from __future__ import (absolute_import, division, print_function)
from ansible.module_utils.basic import AnsibleModule, missing_required_lib
from ansible_collections.infinidat.infinibox.plugins.module_utils.infinibox import (
@api_wrapper
def delete_mapping_to_host(module, system):
    """
    Remove mapping of volume from host. If the either the volume or host
    do not exist, then there should be no mapping to unmap. If unmapping
    generates a key error with 'has no logical units' in its message, then
    the volume is not mapped.  Either case, return changed=False.
    """
    changed = False
    msg = ''
    if not module.check_mode:
        volume = get_volume(module, system)
        host = get_host(module, system)
        volume_name = module.params['volume']
        host_name = module.params['host']
        if volume and host:
            try:
                existing_lun = find_host_lun(host, volume)
                host.unmap_volume(volume)
                changed = True
                msg = f"Volume '{volume_name}' was unmapped from host '{host_name}' freeing lun '{existing_lun}'"
            except KeyError as err:
                if 'has no logical units' not in str(err):
                    module.fail_json(f"Cannot unmap volume '{volume_name}' from host '{host_name}': {str(err)}")
                else:
                    msg = f"Volume '{volume_name}' was not mapped to host '{host_name}' and so unmapping was not executed"
        else:
            msg = f"Either volume '{volume_name}' or host '{host_name}' does not exist. Unmapping was not executed"
    else:
        changed = True
    module.exit_json(msg=msg, changed=changed)