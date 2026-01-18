import boto
from boto.compat import json
from boto.connection import AWSQueryConnection
from boto.regioninfo import RegionInfo
from boto.exception import JSONResponseError
from boto.opsworks import exceptions
def describe_rds_db_instances(self, stack_id, rds_db_instance_arns=None):
    """
        Describes Amazon RDS instances.

        **Required Permissions**: To use this action, an IAM user must
        have a Show, Deploy, or Manage permissions level for the
        stack, or an attached policy that explicitly grants
        permissions. For more information on user permissions, see
        `Managing User Permissions`_.

        :type stack_id: string
        :param stack_id: The stack ID that the instances are registered with.
            The operation returns descriptions of all registered Amazon RDS
            instances.

        :type rds_db_instance_arns: list
        :param rds_db_instance_arns: An array containing the ARNs of the
            instances to be described.

        """
    params = {'StackId': stack_id}
    if rds_db_instance_arns is not None:
        params['RdsDbInstanceArns'] = rds_db_instance_arns
    return self.make_request(action='DescribeRdsDbInstances', body=json.dumps(params))