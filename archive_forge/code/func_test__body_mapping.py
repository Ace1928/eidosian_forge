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
def test__body_mapping(self):

    class Test(resource.Resource):
        x = resource.Body('x')
        y = resource.Body('y')
        z = resource.Body('z')
    self.assertIn('x', Test._body_mapping())
    self.assertIn('y', Test._body_mapping())
    self.assertIn('z', Test._body_mapping())