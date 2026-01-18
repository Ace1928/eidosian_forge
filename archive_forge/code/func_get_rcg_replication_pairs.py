from __future__ import (absolute_import, division, print_function)
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.dellemc.powerflex.plugins.module_utils.storage.dell \
def get_rcg_replication_pairs(self, rcg_id):
    """Get rcg replication pair details
            :param rcg_id: ID of the rcg
            :return: RCG replication pair details
        """
    try:
        rcg_pairs = self.powerflex_conn.replication_consistency_group.get_replication_pairs(rcg_id)
        for rcg_pair in rcg_pairs:
            rcg_pair.pop('links', None)
            rcg_pair['localVolumeName'] = self.get_volume(rcg_pair['localVolumeId'], filter_by_name=False)[0]['name']
            rcg_pair['replicationConsistencyGroupName'] = self.get_rcg(rcg_id=rcg_pair['replicationConsistencyGroupId'])['name']
        return rcg_pairs
    except Exception as e:
        errormsg = 'Failed to get the replication pairs for replication consistency group {0} with error {1}'.format(rcg_id, str(e))
        LOG.error(errormsg)
        self.module.fail_json(msg=errormsg)