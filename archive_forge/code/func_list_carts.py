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
@requires(['DateRangeStart'])
@api_action('CartInfo', 15, 12)
def list_carts(self, request, response, **kw):
    """Returns a list of shopping carts in your Webstore that were last
           updated during the time range that you specify.
        """
    return self._post_request(request, kw, response)