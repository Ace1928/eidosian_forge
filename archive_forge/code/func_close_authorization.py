from collections import abc
import xml.sax
import hashlib
import string
from boto.connection import AWSQueryConnection
from boto.exception import BotoServerError
import boto.mws.exception
import boto.mws.response
from boto.handler import XmlHandler
from boto.compat import filter, map, six, encodebytes
@requires(['AmazonAuthorizationId'])
@api_action('OffAmazonPayments', 10, 1)
def close_authorization(self, request, response, **kw):
    """Closes an authorization.
        """
    return self._post_request(request, kw, response)