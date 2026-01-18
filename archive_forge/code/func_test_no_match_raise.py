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
def test_no_match_raise(self):
    self.assertRaises(exceptions.ResourceNotFound, self.no_results.find, self.cloud.compute, 'name', ignore_missing=False)