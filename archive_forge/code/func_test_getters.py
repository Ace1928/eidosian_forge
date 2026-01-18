from tests.unit import unittest
from boto.exception import BotoServerError, S3CreateError, JSONResponseError
from httpretty import HTTPretty, httprettified
def test_getters(self):
    body = 'This is the body'
    bse = BotoServerError('400', 'Bad Request', body=body)
    self.assertEqual(bse.code, bse.error_code)
    self.assertEqual(bse.message, bse.error_message)