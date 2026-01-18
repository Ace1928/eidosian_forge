from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.dellemc.unity.plugins.module_utils.storage.dell \
def resolve_to_snapschedule_id(self, params):
    """ Get snapshot id for a give snap schedule name
        :param params: snap schedule name or id
        :return: snap schedule id after validation
        """
    try:
        snap_sch_id = None
        snapshot_schedule = {}
        if params['name']:
            snapshot_schedule = utils.UnitySnapScheduleList.get(self.unity_conn._cli, name=params['name'])
        elif params['id']:
            snapshot_schedule = utils.UnitySnapScheduleList.get(self.unity_conn._cli, id=params['id'])
        if snapshot_schedule:
            snap_sch_id = snapshot_schedule.id[0]
        if not snap_sch_id:
            errormsg = ('Failed to find the snapshot schedule id against given name or id: {0}'.format(params['name']), params['id'])
            LOG.error(errormsg)
            self.module.fail_json(msg=errormsg)
        return snap_sch_id
    except Exception as e:
        errormsg = 'Failed to find the snapshot schedules with error {0}'.format(str(e))