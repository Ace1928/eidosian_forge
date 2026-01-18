import boto
from boto.compat import json
from boto.connection import AWSQueryConnection
from boto.regioninfo import RegionInfo
from boto.exception import JSONResponseError
from boto.codedeploy import exceptions
def delete_deployment_group(self, application_name, deployment_group_name):
    """
        Deletes a deployment group.

        :type application_name: string
        :param application_name: The name of an existing AWS CodeDeploy
            application within the AWS user account.

        :type deployment_group_name: string
        :param deployment_group_name: The name of an existing deployment group
            for the specified application.

        """
    params = {'applicationName': application_name, 'deploymentGroupName': deployment_group_name}
    return self.make_request(action='DeleteDeploymentGroup', body=json.dumps(params))