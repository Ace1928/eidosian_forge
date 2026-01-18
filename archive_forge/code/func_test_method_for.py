from boto.mws.connection import MWSConnection, api_call_map, destructure_object
from boto.mws.response import (ResponseElement, GetFeedSubmissionListResult,
from boto.exception import BotoServerError
from tests.compat import unittest
from tests.unit import AWSMockServiceTestCase
from mock import MagicMock
def test_method_for(self):
    self.assertTrue('GetFeedSubmissionList' in api_call_map)
    func = self.service_connection.method_for('GetFeedSubmissionList')
    self.assertTrue(callable(func))
    ideal = self.service_connection.get_feed_submission_list
    self.assertEqual(func, ideal)
    func = self.service_connection.method_for('NotHereNorThere')
    self.assertEqual(func, None)