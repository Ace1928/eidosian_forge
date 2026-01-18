import boto
from boto.compat import json
from boto.connection import AWSQueryConnection
from boto.regioninfo import RegionInfo
from boto.exception import JSONResponseError
from boto.opsworks import exceptions
def describe_time_based_auto_scaling(self, instance_ids):
    """
        Describes time-based auto scaling configurations for specified
        instances.


        You must specify at least one of the parameters.


        **Required Permissions**: To use this action, an IAM user must
        have a Show, Deploy, or Manage permissions level for the
        stack, or an attached policy that explicitly grants
        permissions. For more information on user permissions, see
        `Managing User Permissions`_.

        :type instance_ids: list
        :param instance_ids: An array of instance IDs.

        """
    params = {'InstanceIds': instance_ids}
    return self.make_request(action='DescribeTimeBasedAutoScaling', body=json.dumps(params))