import boto
from boto.compat import json
from boto.connection import AWSQueryConnection
from boto.regioninfo import RegionInfo
from boto.exception import JSONResponseError
from boto.kms import exceptions
from boto.compat import six
import base64
def list_key_policies(self, key_id, limit=None, marker=None):
    """
        Retrieves a list of policies attached to a key.

        :type key_id: string
        :param key_id: Unique identifier of the key. This can be an ARN, an
            alias, or a globally unique identifier.

        :type limit: integer
        :param limit: Specify this parameter only when paginating results to
            indicate the maximum number of policies you want listed in the
            response. If there are additional policies beyond the maximum you
            specify, the `Truncated` response element will be set to `true.`

        :type marker: string
        :param marker: Use this parameter only when paginating results, and
            only in a subsequent request after you've received a response where
            the results are truncated. Set it to the value of the `NextMarker`
            in the response you just received.

        """
    params = {'KeyId': key_id}
    if limit is not None:
        params['Limit'] = limit
    if marker is not None:
        params['Marker'] = marker
    return self.make_request(action='ListKeyPolicies', body=json.dumps(params))