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
@requires(['CreatedAfter'], ['LastUpdatedAfter'])
@requires(['MarketplaceId'])
@exclusive(['CreatedAfter'], ['LastUpdatedAfter'])
@dependent('CreatedBefore', ['CreatedAfter'])
@exclusive(['LastUpdatedAfter'], ['BuyerEmail'], ['SellerOrderId'])
@dependent('LastUpdatedBefore', ['LastUpdatedAfter'])
@exclusive(['CreatedAfter'], ['LastUpdatedBefore'])
@structured_objects('OrderTotal', 'ShippingAddress', 'PaymentExecutionDetail')
@structured_lists('MarketplaceId.Id', 'OrderStatus.Status', 'FulfillmentChannel.Channel', 'PaymentMethod.')
@api_action('Orders', 6, 60)
def list_orders(self, request, response, **kw):
    """Returns a list of orders created or updated during a time
           frame that you specify.
        """
    toggle = set(('FulfillmentChannel.Channel.1', 'OrderStatus.Status.1', 'PaymentMethod.1', 'LastUpdatedAfter', 'LastUpdatedBefore'))
    for do, dont in {'BuyerEmail': toggle.union(['SellerOrderId']), 'SellerOrderId': toggle.union(['BuyerEmail'])}.items():
        if do in kw and any((i in dont for i in kw)):
            message = "Don't include {0} when specifying {1}".format(' or '.join(dont), do)
            raise AssertionError(message)
    return self._post_request(request, kw, response)