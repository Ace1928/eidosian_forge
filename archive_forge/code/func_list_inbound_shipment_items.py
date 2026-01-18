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
@requires(['ShipmentId'], ['LastUpdatedAfter', 'LastUpdatedBefore'])
@api_action('Inbound', 30, 0.5)
def list_inbound_shipment_items(self, request, response, **kw):
    """Returns a list of items in a specified inbound shipment, or a
           list of items that were updated within a specified time frame.
        """
    return self._post_request(request, kw, response)