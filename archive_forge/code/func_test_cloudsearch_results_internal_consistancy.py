from tests.compat import mock, unittest
from httpretty import HTTPretty
import json
import requests
from boto.cloudsearch.search import SearchConnection, SearchServiceException
from boto.compat import six, map
def test_cloudsearch_results_internal_consistancy(self):
    """Check the documents length matches the iterator details"""
    search = SearchConnection(endpoint=HOSTNAME)
    results = search.search(q='Test')
    self.assertEqual(len(results), len(results.docs))