import os
from boto.compat import json
from boto.exception import JSONResponseError
from boto.connection import AWSAuthConnection
from boto.regioninfo import RegionInfo
from boto.awslambda import exceptions
def delete_function(self, function_name):
    """
        Deletes the specified Lambda function code and configuration.

        This operation requires permission for the
        `lambda:DeleteFunction` action.

        :type function_name: string
        :param function_name: The Lambda function to delete.

        """
    uri = '/2014-11-13/functions/{0}'.format(function_name)
    return self.make_request('DELETE', uri, expected_status=204)