from tests.compat import mock, unittest
from httpretty import HTTPretty
import json
import requests
from boto.cloudsearch.search import SearchConnection, SearchServiceException
from boto.compat import six, map
def test_cloudsearch_results_iterator(self):
    """Check the results iterator"""
    search = SearchConnection(endpoint=HOSTNAME)
    results = search.search(q='Test')
    results_correct = iter(['12341', '12342', '12343', '12344', '12345', '12346', '12347'])
    for x in results:
        self.assertEqual(x['id'], next(results_correct))