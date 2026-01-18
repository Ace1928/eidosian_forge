import boto
from boto.compat import json
from boto.connection import AWSQueryConnection
from boto.regioninfo import RegionInfo
from boto.exception import JSONResponseError
from boto.codedeploy import exceptions
def batch_get_deployments(self, deployment_ids=None):
    """
        Gets information about one or more deployments.

        :type deployment_ids: list
        :param deployment_ids: A list of deployment IDs, with multiple
            deployment IDs separated by spaces.

        """
    params = {}
    if deployment_ids is not None:
        params['deploymentIds'] = deployment_ids
    return self.make_request(action='BatchGetDeployments', body=json.dumps(params))