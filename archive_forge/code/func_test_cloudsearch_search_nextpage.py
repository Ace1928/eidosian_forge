from tests.compat import mock, unittest
from httpretty import HTTPretty
import json
import requests
from boto.cloudsearch.search import SearchConnection, SearchServiceException
from boto.compat import six, map
def test_cloudsearch_search_nextpage(self):
    """Check next page query is correct"""
    search = SearchConnection(endpoint=HOSTNAME)
    query1 = search.build_query(q='Test')
    query2 = search.build_query(q='Test')
    results = search(query2)
    self.assertEqual(results.next_page().query.start, query1.start + query1.size)
    self.assertEqual(query1.q, query2.q)