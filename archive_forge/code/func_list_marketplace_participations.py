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
@api_action('Sellers', 15, 60)
def list_marketplace_participations(self, request, response, **kw):
    """Returns a list of marketplaces that the seller submitting
           the request can sell in, and a list of participations that
           include seller-specific information in that marketplace.
        """
    return self._post_request(request, kw, response)