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
@requires(['PackageNumber'])
@api_action('Outbound', 30, 0.5)
def get_package_tracking_details(self, request, response, **kw):
    """Returns delivery tracking information for a package in
           an outbound shipment for a Multi-Channel Fulfillment order.
        """
    return self._post_request(request, kw, response)