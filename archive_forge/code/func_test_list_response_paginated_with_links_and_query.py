import itertools
import json
import logging
from unittest import mock
from keystoneauth1 import adapter
import requests
from openstack import exceptions
from openstack import format
from openstack import resource
from openstack.tests.unit import base
from openstack import utils
def test_list_response_paginated_with_links_and_query(self):
    q_limit = 1
    ids = [1, 2]
    mock_response = mock.Mock()
    mock_response.status_code = 200
    mock_response.links = {}
    mock_response.json.side_effect = [{'resources': [{'id': ids[0]}], 'resources_links': [{'href': 'https://example.com/next-url?limit=%d' % q_limit, 'rel': 'next'}]}, {'resources': [{'id': ids[1]}]}, {'resources': []}]
    self.session.get.return_value = mock_response

    class Test(self.test_class):
        _query_mapping = resource.QueryParameters('limit')
    results = list(Test.list(self.session, paginated=True, limit=q_limit))
    self.assertEqual(2, len(results))
    self.assertEqual(ids[0], results[0].id)
    self.assertEqual(ids[1], results[1].id)
    self.assertEqual(mock.call('base_path', headers={'Accept': 'application/json'}, params={'limit': q_limit}, microversion=None), self.session.get.mock_calls[0])
    self.assertEqual(mock.call('https://example.com/next-url', headers={'Accept': 'application/json'}, params={'limit': [str(q_limit)]}, microversion=None), self.session.get.mock_calls[2])
    self.assertEqual(3, len(self.session.get.call_args_list))
    self.assertIsInstance(results[0], self.test_class)