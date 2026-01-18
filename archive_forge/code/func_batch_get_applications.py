import boto
from boto.compat import json
from boto.connection import AWSQueryConnection
from boto.regioninfo import RegionInfo
from boto.exception import JSONResponseError
from boto.codedeploy import exceptions
def batch_get_applications(self, application_names=None):
    """
        Gets information about one or more applications.

        :type application_names: list
        :param application_names: A list of application names, with multiple
            application names separated by spaces.

        """
    params = {}
    if application_names is not None:
        params['applicationNames'] = application_names
    return self.make_request(action='BatchGetApplications', body=json.dumps(params))