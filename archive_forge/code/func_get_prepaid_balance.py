import urllib
import uuid
from boto.connection import AWSQueryConnection
from boto.fps.exception import ResponseErrorFactory
from boto.fps.response import ResponseFactory
import boto.fps.response
@requires(['PrepaidInstrumentId'])
@api_action()
def get_prepaid_balance(self, action, response, **kw):
    """
        Returns the balance available on the given prepaid instrument.
        """
    return self.get_object(action, kw, response)