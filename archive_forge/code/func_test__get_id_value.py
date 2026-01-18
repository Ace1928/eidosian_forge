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
def test__get_id_value(self):
    value = 'id'
    self.assertEqual(value, resource.Resource._get_id(value))