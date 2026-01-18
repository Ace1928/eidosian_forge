from tests.compat import mock, unittest
from httpretty import HTTPretty
import json
import requests
from boto.cloudsearch.search import SearchConnection, SearchServiceException
from boto.compat import six, map
def test_cloudsearch_facet_constraint_single(self):
    search = SearchConnection(endpoint=HOSTNAME)
    search.search(q='Test', facet_constraints={'author': "'John Smith','Mark Smith'"})
    args = self.get_args(HTTPretty.last_request.raw_requestline)
    self.assertEqual(args[b'facet-author-constraints'], [b"'John Smith','Mark Smith'"])