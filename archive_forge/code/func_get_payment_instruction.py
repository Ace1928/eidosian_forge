import urllib
import uuid
from boto.connection import AWSQueryConnection
from boto.fps.exception import ResponseErrorFactory
from boto.fps.response import ResponseFactory
import boto.fps.response
@requires(['TokenId'])
@api_action()
def get_payment_instruction(self, action, response, **kw):
    """
        Gets the payment instruction of a token.
        """
    return self.get_object(action, kw, response)