import boto
import boto.jsonresponse
from boto.compat import json
from boto.regioninfo import RegionInfo
from boto.connection import AWSQueryConnection
def request_environment_info(self, info_type='tail', environment_id=None, environment_name=None):
    """
        Initiates a request to compile the specified type of
        information of the deployed environment.  Setting the InfoType
        to tail compiles the last lines from the application server log
        files of every Amazon EC2 instance in your environment. Use
        RetrieveEnvironmentInfo to access the compiled information.

        :type info_type: string
        :param info_type: The type of information to request.

        :type environment_id: string
        :param environment_id: The ID of the environment of the
            requested data. If no such environment is found,
            RequestEnvironmentInfo returns an InvalidParameterValue error.
            Condition: You must specify either this or an EnvironmentName, or
            both. If you do not specify either, AWS Elastic Beanstalk returns
            MissingRequiredParameter error.

        :type environment_name: string
        :param environment_name: The name of the environment of the
            requested data. If no such environment is found,
            RequestEnvironmentInfo returns an InvalidParameterValue error.
            Condition: You must specify either this or an EnvironmentId, or
            both. If you do not specify either, AWS Elastic Beanstalk returns
            MissingRequiredParameter error.
        """
    params = {'InfoType': info_type}
    if environment_id:
        params['EnvironmentId'] = environment_id
    if environment_name:
        params['EnvironmentName'] = environment_name
    return self._get_response('RequestEnvironmentInfo', params)