from __future__ import (absolute_import, division, print_function)
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.dellemc.powerflex.plugins.module_utils.storage.dell \
def manage_source_volume(self, snap_pol_details, vol_details, source_volume_element):
    """Adding or removing a source volume
            :param snap_pol_details: Details of the snapshot policy details.
            :param vol_details: Details of the volume.
            :param source_volume_element: The index of the source volume in the
                                          list of volumes to be added/removed.
            :return: Boolean indicating whether volume is added/removed.
        """
    try:
        if self.module.params['source_volume'][source_volume_element]['state'] == 'present' and vol_details['snplIdOfSourceVolume'] != snap_pol_details['id']:
            if not self.module.check_mode:
                snap_pol_details = self.powerflex_conn.snapshot_policy.add_source_volume(snapshot_policy_id=snap_pol_details['id'], volume_id=vol_details['id'])
                LOG.info('Source volume successfully added')
            return True
        elif self.module.params['source_volume'][source_volume_element]['state'] == 'absent' and vol_details['snplIdOfSourceVolume'] == snap_pol_details['id']:
            if not self.module.check_mode:
                snap_pol_details = self.powerflex_conn.snapshot_policy.remove_source_volume(snapshot_policy_id=snap_pol_details['id'], volume_id=vol_details['id'], auto_snap_removal_action=self.module.params['source_volume'][source_volume_element]['auto_snap_removal_action'], detach_locked_auto_snaps=self.module.params['source_volume'][source_volume_element]['detach_locked_auto_snapshots'])
                LOG.info('Source volume successfully removed')
            return True
    except Exception as e:
        error_msg = f'Failed to manage the source volume {vol_details['id']} with error {str(e)}'
        LOG.error(error_msg)
        self.module.fail_json(msg=error_msg)