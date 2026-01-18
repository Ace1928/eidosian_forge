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
@requires(['AmazonOrderReferenceId'])
@api_action('OffAmazonPayments', 10, 1)
def close_order_reference(self, request, response, **kw):
    """Confirms that an order reference has been fulfilled (fully
           or partially) and that you do not expect to create any new
           authorizations on this order reference.
        """
    return self._post_request(request, kw, response)