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
def test_get_no_instance(self):
    sot = resource._BaseComponent('test')
    result = sot.__get__(None, None)
    self.assertIs(sot, result)