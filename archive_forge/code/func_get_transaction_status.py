import urllib
import uuid
from boto.connection import AWSQueryConnection
from boto.fps.exception import ResponseErrorFactory
from boto.fps.response import ResponseFactory
import boto.fps.response
@requires(['TransactionId'])
@api_action()
def get_transaction_status(self, action, response, **kw):
    """
        Gets the latest status of a transaction.
        """
    return self.get_object(action, kw, response)