from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.dellemc.unity.plugins.module_utils.storage.dell \
def create_snapshot_schedule(self, name, rule_dict):
    """Create snapshot schedule.
            :param name: The name of the snapshot schedule
            :param rule_dict: The dict of the rule
            :return: Boolean value to indicate if snapshot schedule created
        """
    try:
        utils.snap_schedule.UnitySnapSchedule.create(cli=self.unity_conn._cli, name=name, rules=[rule_dict])
        return True
    except Exception as e:
        errormsg = 'Create operation of snapshot schedule {0} failed with error {1}'.format(name, str(e))
        LOG.error(errormsg)
        self.module.fail_json(msg=errormsg)