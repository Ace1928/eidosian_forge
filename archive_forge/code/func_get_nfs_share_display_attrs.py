from __future__ import absolute_import, division, print_function
import re
import traceback
from ansible.module_utils.basic import AnsibleModule, missing_required_lib
from ansible_collections.dellemc.unity.plugins.module_utils.storage.dell \
def get_nfs_share_display_attrs(nfs_obj):
    """ Provide nfs share attributes for display

    :param nfs: NFS share obj
    :type nfs: UnityNfsShare
    :return: nfs_share_details
    :rtype: dict
    """
    LOG.info('Getting nfs share details from nfs share object')
    nfs_share_details = nfs_obj._get_properties()
    LOG.info('Updating filesystem details')
    nfs_share_details['filesystem']['UnityFileSystem']['name'] = nfs_obj.filesystem.name
    if 'id' not in nfs_share_details['filesystem']['UnityFileSystem']:
        nfs_share_details['filesystem']['UnityFileSystem']['id'] = nfs_obj.filesystem.id
    LOG.info('Updating nas server details')
    nas_details = nfs_obj.filesystem._get_properties()['nas_server']
    nas_details['UnityNasServer']['name'] = nfs_obj.filesystem.nas_server.name
    nfs_share_details['nas_server'] = nas_details
    if is_nfs_obj_for_snap(nfs_obj):
        LOG.info('Updating snap details')
        nfs_share_details['snap']['UnitySnap']['id'] = nfs_obj.snap.id
        nfs_share_details['snap']['UnitySnap']['name'] = nfs_obj.snap.name
    LOG.info('Successfully updated nfs share details')
    return nfs_share_details