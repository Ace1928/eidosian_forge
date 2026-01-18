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
def test_bulk_create_fail_on_request(self):

    class Test(resource.Resource):
        service = self.service_name
        base_path = self.base_path
        create_method = 'POST'
        allow_create = True
        resources_key = 'tests'
    response = FakeResponse({}, status_code=409)
    response.content = '{"TestError": {"message": "Failed to parse request. Required attribute \'foo\' not specified", "type": "HTTPBadRequest", "detail": ""}}'
    response.reason = 'Bad Request'
    self.session.post.return_value = response
    self.assertRaises(exceptions.ConflictException, Test.bulk_create, self.session, [{'name': 'name'}])