import boto
from boto.compat import json
from boto.connection import AWSQueryConnection
from boto.regioninfo import RegionInfo
from boto.exception import JSONResponseError
from boto.codedeploy import exceptions
def get_application(self, application_name):
    """
        Gets information about an application.

        :type application_name: string
        :param application_name: The name of an existing AWS CodeDeploy
            application within the AWS user account.

        """
    params = {'applicationName': application_name}
    return self.make_request(action='GetApplication', body=json.dumps(params))