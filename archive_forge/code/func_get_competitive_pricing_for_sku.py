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
@requires(['MarketplaceId', 'SellerSKUList'])
@structured_lists('SellerSKUList.SellerSKU')
@api_action('Products', 20, 10, 'GetCompetitivePricingForSKU')
def get_competitive_pricing_for_sku(self, request, response, **kw):
    """Returns the current competitive pricing of a product,
           based on the SellerSKUs and MarketplaceId that you specify.
        """
    return self._post_request(request, kw, response)