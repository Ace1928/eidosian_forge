import boto
from boto.compat import json
from boto.connection import AWSQueryConnection
from boto.regioninfo import RegionInfo
from boto.exception import JSONResponseError
from boto.configservice import exceptions
def put_configuration_recorder(self, configuration_recorder):
    """
        Creates a new configuration recorder to record the resource
        configurations.

        You can use this action to change the role ( `roleARN`) of an
        existing recorder. To change the role, call the action on the
        existing configuration recorder and specify a role.

        :type configuration_recorder: dict
        :param configuration_recorder: The configuration recorder object that
            records each configuration change made to the resources. The
            format should follow:

            {'name': 'myrecorder',
             'roleARN': 'arn:aws:iam::123456789012:role/trusted-aws-config'}

        """
    params = {'ConfigurationRecorder': configuration_recorder}
    return self.make_request(action='PutConfigurationRecorder', body=json.dumps(params))