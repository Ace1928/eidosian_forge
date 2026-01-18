from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.dellemc.unity.plugins.module_utils.storage.dell \
def get_snap_schedule(self, name):
    """Get the instance of a snapshot schedule.
            :param name: The name of the snapshot schedule
            :return: instance of the respective snapshot schedule if exist.
        """
    errormsg = 'Failed to get the snapshot schedule {0} with error {1}'
    try:
        LOG.debug('Attempting to get Snapshot Schedule with name %s', name)
        obj_ss = utils.UnitySnapScheduleList.get(self.unity_conn._cli, name=name)
        if obj_ss and len(obj_ss) > 0:
            LOG.info('Successfully got Snapshot Schedule %s', obj_ss)
            return obj_ss
        else:
            msg = 'Failed to get snapshot schedule with name {0}'.format(name)
            LOG.error(msg)
            self.module.fail_json(msg=msg)
    except Exception as e:
        msg = errormsg.format(name, str(e))
        LOG.error(msg)
        self.module.fail_json(msg=msg)