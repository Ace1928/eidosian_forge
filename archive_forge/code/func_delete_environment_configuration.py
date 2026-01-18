import boto
import boto.jsonresponse
from boto.compat import json
from boto.regioninfo import RegionInfo
from boto.connection import AWSQueryConnection
def delete_environment_configuration(self, application_name, environment_name):
    """
        Deletes the draft configuration associated with the running
        environment.  Updating a running environment with any
        configuration changes creates a draft configuration set. You can
        get the draft configuration using DescribeConfigurationSettings
        while the update is in progress or if the update fails. The
        DeploymentStatus for the draft configuration indicates whether
        the deployment is in process or has failed. The draft
        configuration remains in existence until it is deleted with this
        action.

        :type application_name: string
        :param application_name: The name of the application the
            environment is associated with.

        :type environment_name: string
        :param environment_name: The name of the environment to delete
            the draft configuration from.

        """
    params = {'ApplicationName': application_name, 'EnvironmentName': environment_name}
    return self._get_response('DeleteEnvironmentConfiguration', params)