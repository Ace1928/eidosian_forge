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
def test_list_one_page_response_paginated(self):
    id_value = 1
    mock_response = mock.Mock()
    mock_response.status_code = 200
    mock_response.links = {}
    mock_response.json.return_value = {'resources': [{'id': id_value}]}
    self.session.get.return_value = mock_response
    results = list(self.sot.list(self.session, paginated=True))
    self.assertEqual(1, len(results))
    self.assertEqual(1, len(self.session.get.call_args_list))
    self.assertEqual(id_value, results[0].id)
    self.assertIsInstance(results[0], self.test_class)