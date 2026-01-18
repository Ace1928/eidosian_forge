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
@requires(['CartIdList'])
@structured_lists('CartIdList.CartId')
@api_action('CartInfo', 15, 12)
def get_carts(self, request, response, **kw):
    """Returns shopping carts based on the CartId values that you specify.
        """
    return self._post_request(request, kw, response)