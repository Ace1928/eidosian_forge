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
@requires(['SellerFulfillmentOrderId'])
@api_action('Outbound', 30, 0.5)
def cancel_fulfillment_order(self, request, response, **kw):
    """Requests that Amazon stop attempting to fulfill an existing
           fulfillment order.
        """
    return self._post_request(request, kw, response)