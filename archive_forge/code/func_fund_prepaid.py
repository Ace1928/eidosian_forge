import urllib
import uuid
from boto.connection import AWSQueryConnection
from boto.fps.exception import ResponseErrorFactory
from boto.fps.response import ResponseFactory
import boto.fps.response
@needs_caller_reference
@complex_amounts('FundingAmount')
@requires(['PrepaidInstrumentId', 'FundingAmount.Value', 'SenderTokenId', 'FundingAmount.CurrencyCode'])
@api_action()
def fund_prepaid(self, action, response, **kw):
    """
        Funds the prepaid balance on the given prepaid instrument.
        """
    return self.get_object(action, kw, response)