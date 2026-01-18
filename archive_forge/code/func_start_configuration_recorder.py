import boto
from boto.compat import json
from boto.connection import AWSQueryConnection
from boto.regioninfo import RegionInfo
from boto.exception import JSONResponseError
from boto.configservice import exceptions
def start_configuration_recorder(self, configuration_recorder_name):
    """
        Starts recording configurations of all the resources
        associated with the account.

        You must have created at least one delivery channel to
        successfully start the configuration recorder.

        :type configuration_recorder_name: string
        :param configuration_recorder_name: The name of the recorder object
            that records each configuration change made to the resources.

        """
    params = {'ConfigurationRecorderName': configuration_recorder_name}
    return self.make_request(action='StartConfigurationRecorder', body=json.dumps(params))