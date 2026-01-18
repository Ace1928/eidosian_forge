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
def test__prepare_request_with_resource_key(self):
    key = 'key'

    class Test(resource.Resource):
        base_path = '/something'
        resource_key = key
        body_attr = resource.Body('x')
        header_attr = resource.Header('y')
    body_value = 'body'
    header_value = 'header'
    sot = Test(body_attr=body_value, header_attr=header_value, _synchronized=False)
    result = sot._prepare_request(requires_id=False, prepend_key=True)
    self.assertEqual('/something', result.url)
    self.assertEqual({key: {'x': body_value}}, result.body)
    self.assertEqual({'y': header_value}, result.headers)