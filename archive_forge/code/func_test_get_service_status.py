from boto.mws.connection import MWSConnection, api_call_map, destructure_object
from boto.mws.response import (ResponseElement, GetFeedSubmissionListResult,
from boto.exception import BotoServerError
from tests.compat import unittest
from tests.unit import AWSMockServiceTestCase
from mock import MagicMock
def test_get_service_status(self):
    with self.assertRaises(AttributeError) as err:
        self.service_connection.get_service_status()
    self.assertTrue('products' in str(err.exception))
    self.assertTrue('inventory' in str(err.exception))
    self.assertTrue('feeds' in str(err.exception))