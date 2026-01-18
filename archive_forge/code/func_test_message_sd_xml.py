from tests.unit import unittest
from boto.exception import BotoServerError, S3CreateError, JSONResponseError
from httpretty import HTTPretty, httprettified
def test_message_sd_xml(self):
    xml = '\n<Response>\n  <Errors>\n    <Error>\n      <Code>AuthorizationFailure</Code>\n      <Message>Session does not have permission to perform (sdb:CreateDomain) on resource (arn:aws:sdb:us-east-1:xxxxxxx:domain/test_domain). Contact account owner.</Message>\n      <BoxUsage>0.0055590278</BoxUsage>\n    </Error>\n  </Errors>\n  <RequestID>e73bb2bb-63e3-9cdc-f220-6332de66dbbe</RequestID>\n</Response>'
    bse = BotoServerError('403', 'Forbidden', body=xml)
    self.assertEqual(bse.error_message, 'Session does not have permission to perform (sdb:CreateDomain) on resource (arn:aws:sdb:us-east-1:xxxxxxx:domain/test_domain). Contact account owner.')
    self.assertEqual(bse.error_message, bse.message)
    self.assertEqual(bse.box_usage, '0.0055590278')
    self.assertEqual(bse.error_code, 'AuthorizationFailure')
    self.assertEqual(bse.status, '403')
    self.assertEqual(bse.reason, 'Forbidden')