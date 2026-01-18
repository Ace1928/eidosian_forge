from tests.unit import unittest
from boto.exception import BotoServerError, S3CreateError, JSONResponseError
from httpretty import HTTPretty, httprettified
def test_botoservererror_basics(self):
    bse = BotoServerError('400', 'Bad Request')
    self.assertEqual(bse.status, '400')
    self.assertEqual(bse.reason, 'Bad Request')