import urllib
import uuid
from boto.connection import AWSQueryConnection
from boto.fps.exception import ResponseErrorFactory
from boto.fps.response import ResponseFactory
import boto.fps.response
@needs_caller_reference
@complex_amounts('TransactionAmount')
@requires(['SenderTokenId', 'TransactionAmount.Value', 'TransactionAmount.CurrencyCode'])
@api_action()
def pay(self, action, response, **kw):
    """
        Allows calling applications to move money from a sender to a recipient.
        """
    return self.get_object(action, kw, response)