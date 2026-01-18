import boto
import boto.jsonresponse
from boto.compat import json
from boto.regioninfo import RegionInfo
from boto.connection import AWSQueryConnection
def restart_app_server(self, environment_id=None, environment_name=None):
    """
        Causes the environment to restart the application container
        server running on each Amazon EC2 instance.

        :type environment_id: string
        :param environment_id: The ID of the environment to restart the server
            for.  Condition: You must specify either this or an
            EnvironmentName, or both. If you do not specify either, AWS Elastic
            Beanstalk returns MissingRequiredParameter error.

        :type environment_name: string
        :param environment_name: The name of the environment to restart the
            server for.  Condition: You must specify either this or an
            EnvironmentId, or both. If you do not specify either, AWS Elastic
            Beanstalk returns MissingRequiredParameter error.
        """
    params = {}
    if environment_id:
        params['EnvironmentId'] = environment_id
    if environment_name:
        params['EnvironmentName'] = environment_name
    return self._get_response('RestartAppServer', params)