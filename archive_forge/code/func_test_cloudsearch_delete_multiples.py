from tests.unit import unittest
from httpretty import HTTPretty
from mock import MagicMock
import json
from boto.cloudsearch.document import DocumentServiceConnection
from boto.cloudsearch.document import CommitMismatchError, EncodingError, \
import boto
def test_cloudsearch_delete_multiples(self):
    document = DocumentServiceConnection(endpoint='doc-demo-userdomain.us-east-1.cloudsearch.amazonaws.com')
    document.delete('5', '10')
    document.delete('6', '11')
    document.commit()
    args = json.loads(HTTPretty.last_request.body.decode('utf-8'))
    self.assertEqual(len(args), 2)
    for arg in args:
        self.assertEqual(arg['type'], 'delete')
        if arg['id'] == '5':
            self.assertEqual(arg['version'], '10')
        elif arg['id'] == '6':
            self.assertEqual(arg['version'], '11')
        else:
            self.assertTrue(False)