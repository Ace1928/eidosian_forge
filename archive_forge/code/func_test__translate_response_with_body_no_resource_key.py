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
def test__translate_response_with_body_no_resource_key(self):

    class Test(resource.Resource):
        attr = resource.Body('attr')
    body = {'attr': 'value'}
    response = FakeResponse(body)
    sot = Test()
    sot._filter_component = mock.Mock(side_effect=[body, dict()])
    sot._translate_response(response, has_body=True)
    self.assertEqual('value', sot.attr)
    self.assertEqual(dict(), sot._body.dirty)
    self.assertEqual(dict(), sot._header.dirty)