from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.dellemc.unity.plugins.module_utils.storage.dell \
def get_raid_groups(self, raid_groups):
    """ Get the raid groups for creating pool"""
    try:
        disk_obj = utils.UnityDiskGroup.get(self.conn._cli, _id=raid_groups['disk_group_id'])
        disk_num = raid_groups['disk_num']
        raid_type = raid_groups['raid_type']
        raid_type = self.get_raid_type_enum(raid_type) if raid_type else None
        stripe_width = raid_groups['stripe_width']
        stripe_width = self.get_raid_stripe_width_enum(stripe_width) if stripe_width else None
        raid_group = utils.RaidGroupParameter(disk_group=disk_obj, disk_num=disk_num, raid_type=raid_type, stripe_width=stripe_width)
        raid_groups = [raid_group]
        return raid_groups
    except Exception as e:
        error_message = 'Failed to create storage pool with error: %s' % str(e)
        LOG.error(error_message)
        self.module.fail_json(msg=error_message)