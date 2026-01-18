import boto
from boto.compat import json
from boto.connection import AWSQueryConnection
from boto.regioninfo import RegionInfo
from boto.exception import JSONResponseError
from boto.codedeploy import exceptions
def list_deployment_instances(self, deployment_id, next_token=None, instance_status_filter=None):
    """
        Lists the Amazon EC2 instances for a deployment within the AWS
        user account.

        :type deployment_id: string
        :param deployment_id: The unique ID of a deployment.

        :type next_token: string
        :param next_token: An identifier that was returned from the previous
            list deployment instances call, which can be used to return the
            next set of deployment instances in the list.

        :type instance_status_filter: list
        :param instance_status_filter:
        A subset of instances to list, by status:


        + Pending: Include in the resulting list those instances with pending
              deployments.
        + InProgress: Include in the resulting list those instances with in-
              progress deployments.
        + Succeeded: Include in the resulting list those instances with
              succeeded deployments.
        + Failed: Include in the resulting list those instances with failed
              deployments.
        + Skipped: Include in the resulting list those instances with skipped
              deployments.
        + Unknown: Include in the resulting list those instances with
              deployments in an unknown state.

        """
    params = {'deploymentId': deployment_id}
    if next_token is not None:
        params['nextToken'] = next_token
    if instance_status_filter is not None:
        params['instanceStatusFilter'] = instance_status_filter
    return self.make_request(action='ListDeploymentInstances', body=json.dumps(params))