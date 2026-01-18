from tests.compat import mock, unittest
from httpretty import HTTPretty
import json
import requests
from boto.cloudsearch.search import SearchConnection, SearchServiceException
from boto.compat import six, map
def test_cloudsearch_top_n_multiple(self):
    search = SearchConnection(endpoint=HOSTNAME)
    search.search(q='Test', facet_top_n={'author': 5, 'cat': 10})
    args = self.get_args(HTTPretty.last_request.raw_requestline)
    self.assertEqual(args[b'facet-author-top-n'], [b'5'])
    self.assertEqual(args[b'facet-cat-top-n'], [b'10'])