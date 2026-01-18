import boto
from boto.compat import json
from boto.connection import AWSQueryConnection
from boto.regioninfo import RegionInfo
from boto.exception import JSONResponseError
from boto.opsworks import exceptions
def deregister_instance(self, instance_id):
    """
        Deregister a registered Amazon EC2 or on-premises instance.
        This action removes the instance from the stack and returns it
        to your control. This action can not be used with instances
        that were created with AWS OpsWorks.

        **Required Permissions**: To use this action, an IAM user must
        have a Manage permissions level for the stack or an attached
        policy that explicitly grants permissions. For more
        information on user permissions, see `Managing User
        Permissions`_.

        :type instance_id: string
        :param instance_id: The instance ID.

        """
    params = {'InstanceId': instance_id}
    return self.make_request(action='DeregisterInstance', body=json.dumps(params))