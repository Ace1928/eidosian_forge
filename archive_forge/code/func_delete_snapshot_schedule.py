from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.dellemc.unity.plugins.module_utils.storage.dell \
def delete_snapshot_schedule(self, id):
    """Delete snapshot schedule.
            :param id: The ID of the snapshot schedule
            :return: The boolean value to indicate if snapshot schedule
             deleted
        """
    try:
        obj_schedule = self.return_schedule_instance(id=id)
        obj_schedule.delete()
        return True
    except Exception as e:
        errormsg = 'Delete operation of snapshot schedule id:{0} failed with error {1}'.format(id, str(e))
        LOG.error(errormsg)
        self.module.fail_json(msg=errormsg)