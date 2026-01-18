import boto
from boto.compat import json
from boto.connection import AWSQueryConnection
from boto.regioninfo import RegionInfo
from boto.exception import JSONResponseError
from boto.opsworks import exceptions
def detach_elastic_load_balancer(self, elastic_load_balancer_name, layer_id):
    """
        Detaches a specified Elastic Load Balancing instance from its
        layer.

        **Required Permissions**: To use this action, an IAM user must
        have a Manage permissions level for the stack, or an attached
        policy that explicitly grants permissions. For more
        information on user permissions, see `Managing User
        Permissions`_.

        :type elastic_load_balancer_name: string
        :param elastic_load_balancer_name: The Elastic Load Balancing
            instance's name.

        :type layer_id: string
        :param layer_id: The ID of the layer that the Elastic Load Balancing
            instance is attached to.

        """
    params = {'ElasticLoadBalancerName': elastic_load_balancer_name, 'LayerId': layer_id}
    return self.make_request(action='DetachElasticLoadBalancer', body=json.dumps(params))