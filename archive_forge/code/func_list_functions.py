import os
from boto.compat import json
from boto.exception import JSONResponseError
from boto.connection import AWSAuthConnection
from boto.regioninfo import RegionInfo
from boto.awslambda import exceptions
def list_functions(self, marker=None, max_items=None):
    """
        Returns a list of your Lambda functions. For each function,
        the response includes the function configuration information.
        You must use GetFunction to retrieve the code for your
        function.

        This operation requires permission for the
        `lambda:ListFunctions` action.

        :type marker: string
        :param marker: Optional string. An opaque pagination token returned
            from a previous `ListFunctions` operation. If present, indicates
            where to continue the listing.

        :type max_items: integer
        :param max_items: Optional integer. Specifies the maximum number of AWS
            Lambda functions to return in response. This parameter value must
            be greater than 0.

        """
    uri = '/2014-11-13/functions/'
    params = {}
    headers = {}
    query_params = {}
    if marker is not None:
        query_params['Marker'] = marker
    if max_items is not None:
        query_params['MaxItems'] = max_items
    return self.make_request('GET', uri, expected_status=200, data=json.dumps(params), headers=headers, params=query_params)