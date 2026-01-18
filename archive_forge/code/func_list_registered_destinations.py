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
@requires(['MarketplaceId'])
@api_action('Subscriptions', 25, 0.5)
def list_registered_destinations(self, request, response, **kw):
    """Lists all current destinations that you have registered.
        """
    return self._post_request(request, kw, response)