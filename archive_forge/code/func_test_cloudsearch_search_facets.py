from tests.compat import mock, unittest
from httpretty import HTTPretty
import json
import requests
from boto.cloudsearch.search import SearchConnection, SearchServiceException
from boto.compat import six, map
def test_cloudsearch_search_facets(self):
    search = SearchConnection(endpoint=HOSTNAME)
    results = search.search(q='Test', facet=['tags'])
    self.assertTrue('tags' not in results.facets)
    self.assertEqual(results.facets['animals'], {u'lions': u'1', u'fish': u'2'})