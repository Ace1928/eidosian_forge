from __future__ import (absolute_import, division, print_function)
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.dellemc.powerflex.plugins.module_utils.storage.dell \
def modify_rcg(self, rcg_id, rcg_details):
    rcg_state = self.module.params['rcg_state']
    create_snapshot = self.module.params['create_snapshot']
    rpo = self.module.params['rpo']
    target_volume_access_mode = self.module.params['target_volume_access_mode']
    is_consistent = self.module.params['is_consistent']
    activity_mode = self.module.params['activity_mode']
    new_rcg_name = self.module.params['new_rcg_name']
    changed = False
    pause, freeze = self.get_pause_and_freeze_value()
    if create_snapshot is True:
        changed = self.create_rcg_snapshot(rcg_id)
    if rpo and rcg_details['rpoInSeconds'] and (rpo != rcg_details['rpoInSeconds']):
        changed = self.modify_rpo(rcg_id, rpo)
    if target_volume_access_mode and rcg_details['targetVolumeAccessMode'] != target_volume_access_mode:
        changed = self.modify_target_volume_access_mode(rcg_id, target_volume_access_mode)
    if activity_mode and self.modify_activity_mode(rcg_id, rcg_details, activity_mode):
        changed = True
        rcg_details = self.get_rcg(rcg_id=rcg_details['id'])
    if pause is not None and self.pause_or_resume_rcg(rcg_id, rcg_details, pause, self.module.params['pause_mode']):
        changed = True
    if freeze is not None and self.freeze_or_unfreeze_rcg(rcg_id, rcg_details, freeze):
        changed = True
    if is_consistent is not None and self.set_consistency(rcg_id, rcg_details, is_consistent):
        changed = True
    if new_rcg_name and self.rename_rcg(rcg_id, rcg_details, new_rcg_name):
        changed = True
    if rcg_state == 'sync' and self.sync(rcg_id):
        changed = True
    rcg_action_status = self.perform_rcg_action(rcg_id, rcg_details)
    changed = changed or rcg_action_status
    return changed