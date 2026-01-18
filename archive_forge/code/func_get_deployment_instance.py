import boto
from boto.compat import json
from boto.connection import AWSQueryConnection
from boto.regioninfo import RegionInfo
from boto.exception import JSONResponseError
from boto.codedeploy import exceptions
def get_deployment_instance(self, deployment_id, instance_id):
    """
        Gets information about an Amazon EC2 instance as part of a
        deployment.

        :type deployment_id: string
        :param deployment_id: The unique ID of a deployment.

        :type instance_id: string
        :param instance_id: The unique ID of an Amazon EC2 instance in the
            deployment's deployment group.

        """
    params = {'deploymentId': deployment_id, 'instanceId': instance_id}
    return self.make_request(action='GetDeploymentInstance', body=json.dumps(params))