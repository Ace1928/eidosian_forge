import boto
from boto.compat import json
from boto.connection import AWSQueryConnection
from boto.regioninfo import RegionInfo
from boto.exception import JSONResponseError
from boto.kms import exceptions
from boto.compat import six
import base64
def list_aliases(self, limit=None, marker=None):
    """
        Lists all of the key aliases in the account.

        :type limit: integer
        :param limit: Specify this parameter when paginating results to
            indicate the maximum number of aliases you want in each response.
            If there are additional aliases beyond the maximum you specify, the
            `Truncated` response element will be set to `true.`

        :type marker: string
        :param marker: Use this parameter when paginating results, and only in
            a subsequent request after you've received a response where the
            results are truncated. Set it to the value of the `NextMarker`
            element in the response you just received.

        """
    params = {}
    if limit is not None:
        params['Limit'] = limit
    if marker is not None:
        params['Marker'] = marker
    return self.make_request(action='ListAliases', body=json.dumps(params))