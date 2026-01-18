from tests.compat import mock, unittest
from httpretty import HTTPretty
import json
import requests
from boto.cloudsearch.search import SearchConnection, SearchServiceException
from boto.compat import six, map
def test_cloudsearch_facet_constraint_multiple(self):
    search = SearchConnection(endpoint=HOSTNAME)
    search.search(q='Test', facet_constraints={'author': "'John Smith','Mark Smith'", 'category': "'News','Reviews'"})
    args = self.get_args(HTTPretty.last_request.raw_requestline)
    self.assertEqual(args[b'facet-author-constraints'], [b"'John Smith','Mark Smith'"])
    self.assertEqual(args[b'facet-category-constraints'], [b"'News','Reviews'"])