from __future__ import (absolute_import, division, print_function)
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.dellemc.powerflex.plugins.module_utils.storage.dell \
def pause_snapshot_policy(self, snap_pol_details):
    """Pausing or resuming a snapshot policy.
            :param snap_pol_details: Details of the snapshot policy details.
            :return: Boolean indicating whether snapshot policy is paused/removed or not.
        """
    try:
        if self.module.params['pause'] and snap_pol_details['snapshotPolicyState'] != 'Paused':
            if not self.module.check_mode:
                self.powerflex_conn.snapshot_policy.pause(snapshot_policy_id=snap_pol_details['id'])
                LOG.info('Snapshot policy successfully paused.')
            return True
        elif not self.module.params['pause'] and snap_pol_details['snapshotPolicyState'] == 'Paused':
            if not self.module.check_mode:
                self.powerflex_conn.snapshot_policy.resume(snapshot_policy_id=snap_pol_details['id'])
                LOG.info('Snapshot policy successfully resumed.')
            return True
    except Exception as e:
        error_msg = f'Failed to pause/resume {snap_pol_details['id']} with error {str(e)}'
        LOG.error(error_msg)
        self.module.fail_json(msg=error_msg)