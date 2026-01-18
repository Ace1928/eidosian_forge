from __future__ import absolute_import, division, print_function
import logging
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.dellemc.unity.plugins.module_utils.storage.dell \
def modify_cg(self, cg_name, description, snap_schedule, tiering_policy):
    """Modify consistency group.
            :param cg_name: The name of the consistency group
            :param description: The description of the consistency group
            :param snap_schedule: The name of the snapshot schedule
            :param tiering_policy: The tiering policy that is to be applied to
            consistency group
            :return: The boolean value to indicate if consistency group
             modified
        """
    cg_obj = self.return_cg_instance(cg_name)
    is_snap_schedule_paused = None
    if self.module.params['snap_schedule'] == '':
        is_snap_schedule_paused = False
    if snap_schedule is not None:
        if snap_schedule == '':
            snap_schedule = {'name': None}
        else:
            snap_schedule = {'name': snap_schedule}
    policy_enum = None
    if tiering_policy:
        if utils.TieringPolicyEnum[tiering_policy]:
            policy_enum = utils.TieringPolicyEnum[tiering_policy]
        else:
            errormsg = 'Invalid choice {0} for tiering policy'.format(tiering_policy)
            LOG.error(errormsg)
            self.module.fail_json(msg=errormsg)
    try:
        cg_obj.modify(description=description, snap_schedule=snap_schedule, tiering_policy=policy_enum, is_snap_schedule_paused=is_snap_schedule_paused)
        return True
    except Exception as e:
        errormsg = 'Modify operation of consistency group {0} failed with error {1}'.format(cg_name, str(e))
        LOG.error(errormsg)
        self.module.fail_json(msg=errormsg)