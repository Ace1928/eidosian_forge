import boto
from boto.compat import json
from boto.connection import AWSQueryConnection
from boto.regioninfo import RegionInfo
from boto.exception import JSONResponseError
from boto.codedeploy import exceptions
def stop_deployment(self, deployment_id):
    """
        Attempts to stop an ongoing deployment.

        :type deployment_id: string
        :param deployment_id: The unique ID of a deployment.

        """
    params = {'deploymentId': deployment_id}
    return self.make_request(action='StopDeployment', body=json.dumps(params))