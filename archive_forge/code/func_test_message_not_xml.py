from tests.unit import unittest
from boto.exception import BotoServerError, S3CreateError, JSONResponseError
from httpretty import HTTPretty, httprettified
def test_message_not_xml(self):
    body = 'This is not XML'
    bse = BotoServerError('400', 'Bad Request', body=body)
    self.assertEqual(bse.error_message, 'This is not XML')