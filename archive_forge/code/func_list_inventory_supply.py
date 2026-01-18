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
@requires(['SellerSkus'], ['QueryStartDateTime'])
@structured_lists('SellerSkus.member')
@api_action('Inventory', 30, 0.5)
def list_inventory_supply(self, request, response, **kw):
    """Returns information about the availability of a seller's
           inventory.
        """
    return self._post_request(request, kw, response)