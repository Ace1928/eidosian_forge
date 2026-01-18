import urllib
import uuid
from boto.connection import AWSQueryConnection
from boto.fps.exception import ResponseErrorFactory
from boto.fps.response import ResponseFactory
import boto.fps.response
@requires(['CallerReference'], ['TokenId'])
@api_action()
def get_token_by_caller(self, action, response, **kw):
    """
        Returns the details of a particular token installed by this calling
        application using the subway co-branded UI.
        """
    return self.get_object(action, kw, response)