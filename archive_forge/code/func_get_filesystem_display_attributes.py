from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.dellemc.unity.plugins.module_utils.storage.dell \
def get_filesystem_display_attributes(self, obj_fs):
    """get display filesystem attributes
        :param obj_fs: filesystem instance
        :return: filesystem dict to display
        """
    try:
        obj_fs = obj_fs.update()
        filesystem_details = obj_fs._get_properties()
        filesystem_details['size_total_with_unit'] = utils.convert_size_with_unit(int(filesystem_details['size_total']))
        if obj_fs.pool:
            filesystem_details.update({'pool': {'name': obj_fs.pool.name, 'id': obj_fs.pool.id}})
        if obj_fs.nas_server:
            filesystem_details.update({'nas_server': {'name': obj_fs.nas_server.name, 'id': obj_fs.nas_server.id}})
        snap_list = []
        if obj_fs.has_snap():
            for snap in obj_fs.snapshots:
                d = {'name': snap.name, 'id': snap.id}
                snap_list.append(d)
        filesystem_details['snapshots'] = snap_list
        if obj_fs.storage_resource.snap_schedule:
            filesystem_details['snap_schedule_id'] = obj_fs.storage_resource.snap_schedule.id
            filesystem_details['snap_schedule_name'] = obj_fs.storage_resource.snap_schedule.name
        quota_config_obj = self.get_quota_config_details(obj_fs)
        if quota_config_obj:
            hard_limit = utils.convert_size_with_unit(quota_config_obj.default_hard_limit)
            soft_limit = utils.convert_size_with_unit(quota_config_obj.default_soft_limit)
            grace_period = get_time_with_unit(quota_config_obj.grace_period)
            filesystem_details.update({'quota_config': {'id': quota_config_obj.id, 'default_hard_limit': hard_limit, 'default_soft_limit': soft_limit, 'is_user_quota_enabled': quota_config_obj.is_user_quota_enabled, 'quota_policy': quota_config_obj._get_properties()['quota_policy'], 'grace_period': grace_period}})
        filesystem_details['replication_sessions'] = []
        fs_repl_sessions = self.get_replication_session(obj_fs)
        if fs_repl_sessions:
            filesystem_details['replication_sessions'] = fs_repl_sessions._get_properties()
        return filesystem_details
    except Exception as e:
        errormsg = 'Failed to display the filesystem {0} with error {1}'.format(obj_fs.name, str(e))
        LOG.error(errormsg)
        self.module.fail_json(msg=errormsg)