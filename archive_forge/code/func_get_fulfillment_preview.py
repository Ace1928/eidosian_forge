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
@requires(['Address', 'Items'])
@structured_objects('Address', 'Items')
@api_action('Outbound', 30, 0.5)
def get_fulfillment_preview(self, request, response, **kw):
    """Returns a list of fulfillment order previews based on items
           and shipping speed categories that you specify.
        """
    return self._post_request(request, kw, response)