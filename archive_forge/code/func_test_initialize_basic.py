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
def test_initialize_basic(self):
    body = {'body': 1}
    header = {'header': 2, 'Location': 'somewhere'}
    uri = {'uri': 3}
    computed = {'computed': 4}
    everything = dict(itertools.chain(body.items(), header.items(), uri.items(), computed.items()))
    mock_collect = mock.Mock()
    mock_collect.return_value = (body, header, uri, computed)
    with mock.patch.object(resource.Resource, '_collect_attrs', mock_collect):
        sot = resource.Resource(_synchronized=False, **everything)
        mock_collect.assert_called_once_with(everything)
    self.assertIsNone(sot.location)
    self.assertIsInstance(sot._body, resource._ComponentManager)
    self.assertEqual(body, sot._body.dirty)
    self.assertIsInstance(sot._header, resource._ComponentManager)
    self.assertEqual(header, sot._header.dirty)
    self.assertIsInstance(sot._uri, resource._ComponentManager)
    self.assertEqual(uri, sot._uri.dirty)
    self.assertFalse(sot.allow_create)
    self.assertFalse(sot.allow_fetch)
    self.assertFalse(sot.allow_commit)
    self.assertFalse(sot.allow_delete)
    self.assertFalse(sot.allow_list)
    self.assertFalse(sot.allow_head)
    self.assertEqual('PUT', sot.commit_method)
    self.assertEqual('POST', sot.create_method)