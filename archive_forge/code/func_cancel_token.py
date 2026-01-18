import urllib
import uuid
from boto.connection import AWSQueryConnection
from boto.fps.exception import ResponseErrorFactory
from boto.fps.response import ResponseFactory
import boto.fps.response
@requires(['TokenId'])
@api_action()
def cancel_token(self, action, response, **kw):
    """
        Cancels any token installed by the calling application on its own
        account.
        """
    return self.get_object(action, kw, response)