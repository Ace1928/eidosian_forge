from boto.ec2.ec2object import EC2Object, TaggedEC2Object
from boto.ec2.blockdevicemapping import BlockDeviceMapping
def set_launch_permissions(self, user_ids=None, group_names=None, dry_run=False):
    return self.connection.modify_image_attribute(self.id, 'launchPermission', 'add', user_ids, group_names, dry_run=dry_run)