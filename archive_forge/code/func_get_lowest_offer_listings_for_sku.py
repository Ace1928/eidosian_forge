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
@api_action('Products', 20, 5, 'GetLowestOfferListingsForSKU')
def get_lowest_offer_listings_for_sku(self, request, response, **kw):
    """Returns the lowest price offer listings for a specific
           product by item condition and SellerSKUs.
        """
    return self._post_request(request, kw, response)