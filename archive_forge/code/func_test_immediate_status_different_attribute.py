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
def test_immediate_status_different_attribute(self):
    status = 'loling'
    res = mock.Mock(spec=['id', 'mood'])
    res.mood = status
    result = resource.wait_for_status(self.cloud.compute, res, status, None, interval=1, wait=1, attribute='mood')
    self.assertEqual(res, result)