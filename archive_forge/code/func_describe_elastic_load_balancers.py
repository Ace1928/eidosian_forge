import boto
from boto.compat import json
from boto.connection import AWSQueryConnection
from boto.regioninfo import RegionInfo
from boto.exception import JSONResponseError
from boto.opsworks import exceptions
def describe_elastic_load_balancers(self, stack_id=None, layer_ids=None):
    """
        Describes a stack's Elastic Load Balancing instances.


        You must specify at least one of the parameters.


        **Required Permissions**: To use this action, an IAM user must
        have a Show, Deploy, or Manage permissions level for the
        stack, or an attached policy that explicitly grants
        permissions. For more information on user permissions, see
        `Managing User Permissions`_.

        :type stack_id: string
        :param stack_id: A stack ID. The action describes the stack's Elastic
            Load Balancing instances.

        :type layer_ids: list
        :param layer_ids: A list of layer IDs. The action describes the Elastic
            Load Balancing instances for the specified layers.

        """
    params = {}
    if stack_id is not None:
        params['StackId'] = stack_id
    if layer_ids is not None:
        params['LayerIds'] = layer_ids
    return self.make_request(action='DescribeElasticLoadBalancers', body=json.dumps(params))