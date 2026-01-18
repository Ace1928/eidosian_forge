import boto
from boto.compat import json
from boto.connection import AWSQueryConnection
from boto.regioninfo import RegionInfo
from boto.exception import JSONResponseError
from boto.opsworks import exceptions
def deregister_volume(self, volume_id):
    """
        Deregisters an Amazon EBS volume. The volume can then be
        registered by another stack. For more information, see
        `Resource Management`_.

        **Required Permissions**: To use this action, an IAM user must
        have a Manage permissions level for the stack, or an attached
        policy that explicitly grants permissions. For more
        information on user permissions, see `Managing User
        Permissions`_.

        :type volume_id: string
        :param volume_id: The volume ID.

        """
    params = {'VolumeId': volume_id}
    return self.make_request(action='DeregisterVolume', body=json.dumps(params))