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
@requires(['NextToken'])
@api_action('Inbound', 30, 0.5)
def list_inbound_shipments_by_next_token(self, request, response, **kw):
    """Returns the next page of inbound shipments using the NextToken
           parameter.
        """
    return self._post_request(request, kw, response)