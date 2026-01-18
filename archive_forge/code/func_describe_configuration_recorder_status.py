import boto
from boto.compat import json
from boto.connection import AWSQueryConnection
from boto.regioninfo import RegionInfo
from boto.exception import JSONResponseError
from boto.configservice import exceptions
def describe_configuration_recorder_status(self, configuration_recorder_names=None):
    """
        Returns the current status of the specified configuration
        recorder. If a configuration recorder is not specified, this
        action returns the status of all configuration recorder
        associated with the account.

        :type configuration_recorder_names: list
        :param configuration_recorder_names: The name(s) of the configuration
            recorder. If the name is not specified, the action returns the
            current status of all the configuration recorders associated with
            the account.

        """
    params = {}
    if configuration_recorder_names is not None:
        params['ConfigurationRecorderNames'] = configuration_recorder_names
    return self.make_request(action='DescribeConfigurationRecorderStatus', body=json.dumps(params))