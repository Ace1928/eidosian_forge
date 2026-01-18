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
@requires(['AmazonOrderReferenceId', 'OrderReferenceAttributes'])
@structured_objects('OrderReferenceAttributes')
@api_action('OffAmazonPayments', 10, 1)
def set_order_reference_details(self, request, response, **kw):
    """Sets order reference details such as the order total and a
           description for the order.
        """
    return self._post_request(request, kw, response)