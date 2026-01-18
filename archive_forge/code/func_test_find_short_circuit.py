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
def test_find_short_circuit(self):
    value = 1

    class Test(resource.Resource):

        @classmethod
        def existing(cls, **kwargs):
            mock_match = mock.Mock()
            mock_match.fetch.return_value = value
            return mock_match
    result = Test.find(self.cloud.compute, 'name')
    self.assertEqual(result, value)