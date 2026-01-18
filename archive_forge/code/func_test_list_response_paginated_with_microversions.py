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
def test_list_response_paginated_with_microversions(self):

    class Test(resource.Resource):
        service = self.service_name
        base_path = self.base_path
        resources_key = 'resources'
        allow_list = True
        _max_microversion = '1.42'
    ids = [1, 2]
    mock_response = mock.Mock()
    mock_response.status_code = 200
    mock_response.links = {}
    mock_response.json.return_value = {'resources': [{'id': ids[0]}], 'resources_links': [{'href': 'https://example.com/next-url', 'rel': 'next'}]}
    mock_response2 = mock.Mock()
    mock_response2.status_code = 200
    mock_response2.links = {}
    mock_response2.json.return_value = {'resources': [{'id': ids[1]}]}
    self.session.get.side_effect = [mock_response, mock_response2]
    results = list(Test.list(self.session, paginated=True))
    self.assertEqual(2, len(results))
    self.assertEqual(ids[0], results[0].id)
    self.assertEqual(ids[1], results[1].id)
    self.assertEqual(mock.call('base_path', headers={'Accept': 'application/json'}, params={}, microversion='1.42'), self.session.get.mock_calls[0])
    self.assertEqual(mock.call('https://example.com/next-url', headers={'Accept': 'application/json'}, params={}, microversion='1.42'), self.session.get.mock_calls[1])
    self.assertEqual(2, len(self.session.get.call_args_list))
    self.assertIsInstance(results[0], Test)
    self.assertEqual('1.42', results[0].microversion)