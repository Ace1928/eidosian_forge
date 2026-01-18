import urllib
import uuid
from boto.connection import AWSQueryConnection
from boto.fps.exception import ResponseErrorFactory
from boto.fps.response import ResponseFactory
import boto.fps.response
@requires(['SubscriptionId'])
@api_action()
def get_subscription_details(self, action, response, **kw):
    """
        Returns the details of Subscription for a given subscriptionID.
        """
    return self.get_object(action, kw, response)