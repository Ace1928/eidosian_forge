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
def test_head_base_path(self):
    result = self.sot.head(self.session, base_path='dummy')
    self.sot._prepare_request.assert_called_once_with(base_path='dummy')
    self.session.head.assert_called_once_with(self.request.url, microversion=None)
    self.assertIsNone(self.sot.microversion)
    self.sot._translate_response.assert_called_once_with(self.response, has_body=False)
    self.assertEqual(result, self.sot)