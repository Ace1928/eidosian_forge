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
@requires(['AmazonOrderId'])
@structured_lists('AmazonOrderId.Id')
@api_action('Orders', 6, 60)
def get_order(self, request, response, **kw):
    """Returns an order for each AmazonOrderId that you specify.
        """
    return self._post_request(request, kw, response)