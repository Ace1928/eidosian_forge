import urllib
import uuid
from boto.connection import AWSQueryConnection
from boto.fps.exception import ResponseErrorFactory
from boto.fps.response import ResponseFactory
import boto.fps.response
@requires(['StartDate'])
@api_action()
def get_account_activity(self, action, response, **kw):
    """
        Returns transactions for a given date range.
        """
    return self.get_object(action, kw, response)