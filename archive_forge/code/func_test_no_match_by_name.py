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
def test_no_match_by_name(self):
    the_name = 'Brian'
    match = mock.Mock(spec=resource.Resource)
    match.name = the_name
    result = resource.Resource._get_one_match('Richard', [match])
    self.assertIsNone(result, match)