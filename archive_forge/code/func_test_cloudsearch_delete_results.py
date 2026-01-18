from tests.unit import unittest
from httpretty import HTTPretty
from mock import MagicMock
import json
from boto.cloudsearch.document import DocumentServiceConnection
from boto.cloudsearch.document import CommitMismatchError, EncodingError, \
import boto
def test_cloudsearch_delete_results(self):
    """
        Check that the result of a single document deletion is parsed properly.
        """
    document = DocumentServiceConnection(endpoint='doc-demo-userdomain.us-east-1.cloudsearch.amazonaws.com')
    document.delete('5', '10')
    doc = document.commit()
    self.assertEqual(doc.status, 'success')
    self.assertEqual(doc.adds, 0)
    self.assertEqual(doc.deletes, 1)