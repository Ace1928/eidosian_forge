import urllib
import uuid
from boto.connection import AWSQueryConnection
from boto.fps.exception import ResponseErrorFactory
from boto.fps.response import ResponseFactory
import boto.fps.response
@requires(['SubscriptionId'])
@api_action()
def get_transactions_for_subscription(self, action, response, **kw):
    """
        Returns the transactions for a given subscriptionID.
        """
    return self.get_object(action, kw, response)