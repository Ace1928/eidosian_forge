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
def test_list_multi_page_response_paginated(self):
    ids = [1, 2]
    resp1 = mock.Mock()
    resp1.status_code = 200
    resp1.links = {}
    resp1.json.return_value = {'resources': [{'id': ids[0]}], 'resources_links': [{'href': 'https://example.com/next-url', 'rel': 'next'}]}
    resp2 = mock.Mock()
    resp2.status_code = 200
    resp2.links = {}
    resp2.json.return_value = {'resources': [{'id': ids[1]}], 'resources_links': [{'href': 'https://example.com/next-url', 'rel': 'next'}]}
    resp3 = mock.Mock()
    resp3.status_code = 200
    resp3.links = {}
    resp3.json.return_value = {'resources': []}
    self.session.get.side_effect = [resp1, resp2, resp3]
    results = self.sot.list(self.session, paginated=True)
    result0 = next(results)
    self.assertEqual(result0.id, ids[0])
    self.session.get.assert_called_with(self.base_path, headers={'Accept': 'application/json'}, params={}, microversion=None)
    result1 = next(results)
    self.assertEqual(result1.id, ids[1])
    self.session.get.assert_called_with('https://example.com/next-url', headers={'Accept': 'application/json'}, params={}, microversion=None)
    self.assertRaises(StopIteration, next, results)
    self.session.get.assert_called_with('https://example.com/next-url', headers={'Accept': 'application/json'}, params={}, microversion=None)