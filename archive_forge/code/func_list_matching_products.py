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
@requires(['MarketplaceId', 'Query'])
@api_action('Products', 20, 20)
def list_matching_products(self, request, response, **kw):
    """Returns a list of products and their attributes, ordered
           by relevancy, based on a search query that you specify.
        """
    return self._post_request(request, kw, response)