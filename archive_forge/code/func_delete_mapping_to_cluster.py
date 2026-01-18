from __future__ import (absolute_import, division, print_function)
from ansible.module_utils.basic import AnsibleModule, missing_required_lib
from ansible_collections.infinidat.infinibox.plugins.module_utils.infinibox import (
@api_wrapper
def delete_mapping_to_cluster(module, system):
    """
    Remove mapping of volume from cluster. If the either the volume or cluster
    do not exist, then there should be no mapping to unmap. If unmapping
    generates a key error with 'has no logical units' in its message, then
    the volume is not mapped.  Either case, return changed=False.
    """
    changed = False
    msg = ''
    if not module.check_mode:
        volume = get_volume(module, system)
        cluster = get_cluster(module, system)
        volume_name = module.params['volume']
        cluster_name = module.params['cluster']
        if volume and cluster:
            try:
                existing_lun = find_cluster_lun(cluster, volume)
                cluster.unmap_volume(volume)
                changed = True
                msg = f"Volume '{volume_name}' was unmapped from cluster '{cluster_name}' freeing lun '{existing_lun}'"
            except KeyError as err:
                if 'has no logical units' not in str(err):
                    msg = f"Cannot unmap volume '{volume_name}' from cluster '{cluster_name}': {str(err)}"
                    module.fail_json(msg=msg)
                else:
                    msg = f"Volume '{volume_name}' was not mapped to cluster '{cluster_name}' and so unmapping was not executed"
        else:
            msg = f"Either volume '{volume_name}' or cluster '{cluster_name}' does not exist. Unmapping was not executed"
    else:
        changed = True
    module.exit_json(msg=msg, changed=changed)