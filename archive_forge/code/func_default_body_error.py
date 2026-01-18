from boto.mws.connection import MWSConnection, api_call_map, destructure_object
from boto.mws.response import (ResponseElement, GetFeedSubmissionListResult,
from boto.exception import BotoServerError
from tests.compat import unittest
from tests.unit import AWSMockServiceTestCase
from mock import MagicMock
def default_body_error(self):
    return b'<?xml version="1.0" encoding="UTF-8"?>\n<ErrorResponse xmlns="http://mws.amazonaws.com/doc/2009-01-01/">\n  <!--1 or more repetitions:-->\n  <Error>\n    <Type>Sender</Type>\n    <Code>string</Code>\n    <Message>string</Message>\n    <Detail>\n      <!--You may enter ANY elements at this point-->\n      <AnyElement xmlns=""/>\n    </Detail>\n  </Error>\n  <RequestId>string</RequestId>\n</ErrorResponse>'