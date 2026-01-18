from tests.compat import mock, unittest
from httpretty import HTTPretty
import json
import requests
from boto.cloudsearch.search import SearchConnection, SearchServiceException
from boto.compat import six, map
def test_cloudsearch_qsearch(self):
    search = SearchConnection(endpoint=HOSTNAME)
    search.search(q='Test')
    args = self.get_args(HTTPretty.last_request.raw_requestline)
    self.assertEqual(args[b'q'], [b'Test'])
    self.assertEqual(args[b'start'], [b'0'])
    self.assertEqual(args[b'size'], [b'10'])