from tests.unit import unittest
from boto.exception import BotoServerError, S3CreateError, JSONResponseError
from httpretty import HTTPretty, httprettified
def test_message_elb_xml(self):
    xml = '\n<ErrorResponse xmlns="http://elasticloadbalancing.amazonaws.com/doc/2011-11-15/">\n  <Error>\n    <Type>Sender</Type>\n    <Code>LoadBalancerNotFound</Code>\n    <Message>Cannot find Load Balancer webapp-balancer2</Message>\n  </Error>\n  <RequestId>093f80d0-4473-11e1-9234-edce8ec08e2d</RequestId>\n</ErrorResponse>'
    bse = BotoServerError('400', 'Bad Request', body=xml)
    self.assertEqual(bse.error_message, 'Cannot find Load Balancer webapp-balancer2')
    self.assertEqual(bse.error_message, bse.message)
    self.assertEqual(bse.request_id, '093f80d0-4473-11e1-9234-edce8ec08e2d')
    self.assertEqual(bse.error_code, 'LoadBalancerNotFound')
    self.assertEqual(bse.status, '400')
    self.assertEqual(bse.reason, 'Bad Request')