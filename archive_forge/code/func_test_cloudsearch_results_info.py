from tests.compat import mock, unittest
from httpretty import HTTPretty
import json
import requests
from boto.cloudsearch.search import SearchConnection, SearchServiceException
from boto.compat import six, map
def test_cloudsearch_results_info(self):
    """Check num_pages_needed is calculated correctly"""
    search = SearchConnection(endpoint=HOSTNAME)
    results = search.search(q='Test')
    self.assertEqual(results.num_pages_needed, 3.0)