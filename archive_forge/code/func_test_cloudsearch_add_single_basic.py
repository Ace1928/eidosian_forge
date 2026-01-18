from tests.unit import unittest
from httpretty import HTTPretty
from mock import MagicMock
import json
from boto.cloudsearch.document import DocumentServiceConnection
from boto.cloudsearch.document import CommitMismatchError, EncodingError, \
import boto
def test_cloudsearch_add_single_basic(self):
    """
        Check that a simple add document sends correct document metadata to
        AWS.
        """
    document = DocumentServiceConnection(endpoint='doc-demo-userdomain.us-east-1.cloudsearch.amazonaws.com')
    document.add('1234', 10, {'id': '1234', 'title': 'Title 1', 'category': ['cat_a', 'cat_b', 'cat_c']})
    document.commit()
    args = json.loads(HTTPretty.last_request.body.decode('utf-8'))[0]
    self.assertEqual(args['id'], '1234')
    self.assertEqual(args['version'], 10)
    self.assertEqual(args['type'], 'add')