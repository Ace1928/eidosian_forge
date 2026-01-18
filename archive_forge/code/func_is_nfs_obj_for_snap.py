from __future__ import absolute_import, division, print_function
import re
import traceback
from ansible.module_utils.basic import AnsibleModule, missing_required_lib
from ansible_collections.dellemc.unity.plugins.module_utils.storage.dell \
def is_nfs_obj_for_snap(nfs_obj):
    """ Check whether the nfs_obj if for snapshot

    :param nfs_obj: NFS share object
    :return: True if nfs_obj is of snapshot type
    :rtype: bool
    """
    if nfs_obj.type == utils.NFSTypeEnum.NFS_SNAPSHOT:
        return True
    return False