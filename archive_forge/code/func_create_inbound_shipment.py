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
@requires(['ShipmentId', 'InboundShipmentHeader', 'InboundShipmentItems'])
@structured_objects('InboundShipmentHeader', 'InboundShipmentItems')
@api_action('Inbound', 30, 0.5)
def create_inbound_shipment(self, request, response, **kw):
    """Creates an inbound shipment.
        """
    return self._post_request(request, kw, response)