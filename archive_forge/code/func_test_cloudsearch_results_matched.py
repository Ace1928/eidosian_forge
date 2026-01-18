from tests.compat import mock, unittest
from httpretty import HTTPretty
import json
import requests
from boto.cloudsearch.search import SearchConnection, SearchServiceException
from boto.compat import six, map
def test_cloudsearch_results_matched(self):
    """
        Check that information objects are passed back through the API
        correctly.
        """
    search = SearchConnection(endpoint=HOSTNAME)
    query = search.build_query(q='Test')
    results = search(query)
    self.assertEqual(results.search_service, search)
    self.assertEqual(results.query, query)