from __future__ import (absolute_import, division, print_function)
from ansible.module_utils.basic import AnsibleModule, missing_required_lib
from ansible_collections.infinidat.infinibox.plugins.module_utils.infinibox import (
def find_cluster_lun(cluster, volume):
    """ Find a cluster's LUN """
    found_lun = None
    luns = cluster.get_luns()
    for lun in luns:
        if lun.volume == volume:
            found_lun = lun.lun
    return found_lun