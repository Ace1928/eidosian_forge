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
def test_list_multi_page_inferred_additional(self):
    ids = [1, 2, 3]
    resp1 = mock.Mock()
    resp1.status_code = 200
    resp1.links = {}
    resp1.json.return_value = {'resources': [{'id': ids[0]}, {'id': ids[1]}]}
    resp2 = mock.Mock()
    resp2.status_code = 200
    resp2.links = {}
    resp2.json.return_value = {'resources': [{'id': ids[2]}]}
    self.session.get.side_effect = [resp1, resp2]
    results = self.sot.list(self.session, limit=2, paginated=True)
    result0 = next(results)
    self.assertEqual(result0.id, ids[0])
    result1 = next(results)
    self.assertEqual(result1.id, ids[1])
    self.session.get.assert_called_with(self.base_path, headers={'Accept': 'application/json'}, params={'limit': 2}, microversion=None)
    result2 = next(results)
    self.assertEqual(result2.id, ids[2])
    self.session.get.assert_called_with(self.base_path, headers={'Accept': 'application/json'}, params={'limit': 2, 'marker': 2}, microversion=None)
    self.assertRaises((StopIteration, RuntimeError), next, results)
    self.assertEqual(3, len(self.session.get.call_args_list))