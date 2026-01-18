import boto
from boto.compat import json
from boto.connection import AWSQueryConnection
from boto.regioninfo import RegionInfo
from boto.exception import JSONResponseError
from boto.configservice import exceptions
def describe_configuration_recorders(self, configuration_recorder_names=None):
    """
        Returns the name of one or more specified configuration
        recorders. If the recorder name is not specified, this action
        returns the names of all the configuration recorders
        associated with the account.

        :type configuration_recorder_names: list
        :param configuration_recorder_names: A list of configuration recorder
            names.

        """
    params = {}
    if configuration_recorder_names is not None:
        params['ConfigurationRecorderNames'] = configuration_recorder_names
    return self.make_request(action='DescribeConfigurationRecorders', body=json.dumps(params))