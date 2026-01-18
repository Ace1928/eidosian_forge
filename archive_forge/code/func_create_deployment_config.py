import boto
from boto.compat import json
from boto.connection import AWSQueryConnection
from boto.regioninfo import RegionInfo
from boto.exception import JSONResponseError
from boto.codedeploy import exceptions
def create_deployment_config(self, deployment_config_name, minimum_healthy_hosts=None):
    """
        Creates a new deployment configuration.

        :type deployment_config_name: string
        :param deployment_config_name: The name of the deployment configuration
            to create.

        :type minimum_healthy_hosts: dict
        :param minimum_healthy_hosts: The minimum number of healthy instances
            that should be available at any time during the deployment. There
            are two parameters expected in the input: type and value.
        The type parameter takes either of the following values:


        + HOST_COUNT: The value parameter represents the minimum number of
              healthy instances, as an absolute value.
        + FLEET_PERCENT: The value parameter represents the minimum number of
              healthy instances, as a percentage of the total number of instances
              in the deployment. If you specify FLEET_PERCENT, then at the start
              of the deployment AWS CodeDeploy converts the percentage to the
              equivalent number of instances and rounds fractional instances up.


        The value parameter takes an integer.

        For example, to set a minimum of 95% healthy instances, specify a type
            of FLEET_PERCENT and a value of 95.

        """
    params = {'deploymentConfigName': deployment_config_name}
    if minimum_healthy_hosts is not None:
        params['minimumHealthyHosts'] = minimum_healthy_hosts
    return self.make_request(action='CreateDeploymentConfig', body=json.dumps(params))