import boto
from boto.compat import json
from boto.connection import AWSQueryConnection
from boto.regioninfo import RegionInfo
from boto.exception import JSONResponseError
from boto.opsworks import exceptions
def describe_elastic_ips(self, instance_id=None, stack_id=None, ips=None):
    """
        Describes `Elastic IP addresses`_.


        You must specify at least one of the parameters.


        **Required Permissions**: To use this action, an IAM user must
        have a Show, Deploy, or Manage permissions level for the
        stack, or an attached policy that explicitly grants
        permissions. For more information on user permissions, see
        `Managing User Permissions`_.

        :type instance_id: string
        :param instance_id: The instance ID. If you include this parameter,
            `DescribeElasticIps` returns a description of the Elastic IP
            addresses associated with the specified instance.

        :type stack_id: string
        :param stack_id: A stack ID. If you include this parameter,
            `DescribeElasticIps` returns a description of the Elastic IP
            addresses that are registered with the specified stack.

        :type ips: list
        :param ips: An array of Elastic IP addresses to be described. If you
            include this parameter, `DescribeElasticIps` returns a description
            of the specified Elastic IP addresses. Otherwise, it returns a
            description of every Elastic IP address.

        """
    params = {}
    if instance_id is not None:
        params['InstanceId'] = instance_id
    if stack_id is not None:
        params['StackId'] = stack_id
    if ips is not None:
        params['Ips'] = ips
    return self.make_request(action='DescribeElasticIps', body=json.dumps(params))