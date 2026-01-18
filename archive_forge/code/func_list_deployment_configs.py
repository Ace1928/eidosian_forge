import boto
from boto.compat import json
from boto.connection import AWSQueryConnection
from boto.regioninfo import RegionInfo
from boto.exception import JSONResponseError
from boto.codedeploy import exceptions
def list_deployment_configs(self, next_token=None):
    """
        Lists the deployment configurations within the AWS user
        account.

        :type next_token: string
        :param next_token: An identifier that was returned from the previous
            list deployment configurations call, which can be used to return
            the next set of deployment configurations in the list.

        """
    params = {}
    if next_token is not None:
        params['nextToken'] = next_token
    return self.make_request(action='ListDeploymentConfigs', body=json.dumps(params))