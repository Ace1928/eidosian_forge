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
def test__header_mapping(self):

    class Test(resource.Resource):
        x = resource.Header('x')
        y = resource.Header('y')
        z = resource.Header('z')
    self.assertIn('x', Test._header_mapping())
    self.assertIn('y', Test._header_mapping())
    self.assertIn('z', Test._header_mapping())