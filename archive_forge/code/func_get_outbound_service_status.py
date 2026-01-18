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
@api_action('Outbound', 2, 300, 'GetServiceStatus')
def get_outbound_service_status(self, request, response, **kw):
    """Returns the operational status of the Fulfillment Outbound
           API section.
        """
    return self._post_request(request, kw, response)