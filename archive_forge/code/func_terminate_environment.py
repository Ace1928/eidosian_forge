import boto
import boto.jsonresponse
from boto.compat import json
from boto.regioninfo import RegionInfo
from boto.connection import AWSQueryConnection
def terminate_environment(self, environment_id=None, environment_name=None, terminate_resources=None):
    """Terminates the specified environment.

        :type environment_id: string
        :param environment_id: The ID of the environment to terminate.
            Condition: You must specify either this or an EnvironmentName, or
            both.  If you do not specify either, AWS Elastic Beanstalk returns
            MissingRequiredParameter error.

        :type environment_name: string
        :param environment_name: The name of the environment to terminate.
            Condition: You must specify either this or an EnvironmentId, or
            both.  If you do not specify either, AWS Elastic Beanstalk returns
            MissingRequiredParameter error.

        :type terminate_resources: boolean
        :param terminate_resources: Indicates whether the associated AWS
            resources should shut down when the environment is terminated:
            true: (default) The user AWS resources (for example, the Auto
            Scaling group, LoadBalancer, etc.) are terminated along with the
            environment.  false: The environment is removed from the AWS
            Elastic Beanstalk but the AWS resources continue to operate.  For
            more information, see the  AWS Elastic Beanstalk User Guide.
            Default: true  Valid Values: true | false

        :raises: InsufficientPrivilegesException
        """
    params = {}
    if environment_id:
        params['EnvironmentId'] = environment_id
    if environment_name:
        params['EnvironmentName'] = environment_name
    if terminate_resources:
        params['TerminateResources'] = self._encode_bool(terminate_resources)
    return self._get_response('TerminateEnvironment', params)