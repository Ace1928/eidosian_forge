import urllib
import uuid
from boto.connection import AWSQueryConnection
from boto.fps.exception import ResponseErrorFactory
from boto.fps.response import ResponseFactory
import boto.fps.response
@complex_amounts('TransactionAmount')
@requires(['ReserveTransactionId', 'TransactionAmount.Value', 'TransactionAmount.CurrencyCode'])
@api_action()
def settle(self, action, response, **kw):
    """
        The Settle API is used in conjunction with the Reserve API and is used
        to settle previously reserved transaction.
        """
    return self.get_object(action, kw, response)