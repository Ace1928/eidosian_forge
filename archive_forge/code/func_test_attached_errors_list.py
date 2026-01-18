from boto.cloudsearch2.domain import Domain
from boto.cloudsearch2.layer1 import CloudSearchConnection
from tests.unit import unittest, AWSMockServiceTestCase
from httpretty import HTTPretty
from mock import MagicMock
import json
from boto.cloudsearch2.document import DocumentServiceConnection
from boto.cloudsearch2.document import CommitMismatchError, EncodingError, \
import boto
from tests.unit.cloudsearch2 import DEMO_DOMAIN_DATA
def test_attached_errors_list(self):
    document = DocumentServiceConnection(endpoint='doc-demo-userdomain.us-east-1.cloudsearch.amazonaws.com')
    document.add('1234', {'id': '1234', 'title': 'Title 1', 'category': ['cat_a', 'cat_b', 'cat_c']})
    try:
        document.commit()
        self.assertTrue(False)
    except CommitMismatchError as e:
        self.assertTrue(hasattr(e, 'errors'))
        self.assertIsInstance(e.errors, list)
        self.assertEquals(e.errors[0], self.response['errors'][0].get('message'))