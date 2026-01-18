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
def test__prepare_request_missing_id(self):
    sot = resource.Resource(id=None)
    self.assertRaises(exceptions.InvalidRequest, sot._prepare_request, requires_id=True)