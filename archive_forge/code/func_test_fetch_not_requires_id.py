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
def test_fetch_not_requires_id(self):
    result = self.sot.fetch(self.session, False)
    self.sot._prepare_request.assert_called_once_with(requires_id=False, base_path=None)
    self.session.get.assert_called_once_with(self.request.url, microversion=None, params={}, skip_cache=False)
    self.sot._translate_response.assert_called_once_with(self.response)
    self.assertEqual(result, self.sot)