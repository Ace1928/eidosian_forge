import urllib
import uuid
from boto.connection import AWSQueryConnection
from boto.fps.exception import ResponseErrorFactory
from boto.fps.response import ResponseFactory
import boto.fps.response
@api_action()
def get_total_prepaid_liability(self, action, response):
    """
        Returns the total liability held by the given account corresponding to
        all the prepaid instruments owned by the account.
        """
    return self.get_object(action, {}, response)