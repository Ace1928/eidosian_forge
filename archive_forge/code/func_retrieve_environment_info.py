import boto
import boto.jsonresponse
from boto.compat import json
from boto.regioninfo import RegionInfo
from boto.connection import AWSQueryConnection
def retrieve_environment_info(self, info_type='tail', environment_id=None, environment_name=None):
    """
        Retrieves the compiled information from a RequestEnvironmentInfo
        request.

        :type info_type: string
        :param info_type: The type of information to retrieve.

        :type environment_id: string
        :param environment_id: The ID of the data's environment. If no such
            environment is found, returns an InvalidParameterValue error.
            Condition: You must specify either this or an EnvironmentName, or
            both.  If you do not specify either, AWS Elastic Beanstalk returns
            MissingRequiredParameter error.

        :type environment_name: string
        :param environment_name: The name of the data's environment. If no such
            environment is found, returns an InvalidParameterValue error.
            Condition: You must specify either this or an EnvironmentId, or
            both.  If you do not specify either, AWS Elastic Beanstalk returns
            MissingRequiredParameter error.
        """
    params = {'InfoType': info_type}
    if environment_id:
        params['EnvironmentId'] = environment_id
    if environment_name:
        params['EnvironmentName'] = environment_name
    return self._get_response('RetrieveEnvironmentInfo', params)