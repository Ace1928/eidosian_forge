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
def test_implementations(self):
    self.assertEqual('_body', resource.Body.key)
    self.assertEqual('_header', resource.Header.key)
    self.assertEqual('_uri', resource.URI.key)