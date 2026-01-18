import urllib
import uuid
from boto.connection import AWSQueryConnection
from boto.fps.exception import ResponseErrorFactory
from boto.fps.response import ResponseFactory
import boto.fps.response
@needs_caller_reference
@requires(['PaymentInstruction', 'TokenType'])
@api_action()
def install_payment_instruction(self, action, response, **kw):
    """
        Installs a payment instruction for caller.
        """
    return self.get_object(action, kw, response)