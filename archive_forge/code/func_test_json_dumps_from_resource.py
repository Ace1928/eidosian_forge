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
def test_json_dumps_from_resource(self):

    class Test(resource.Resource):
        foo = resource.Body('foo_remote')
    res = Test(foo='bar')
    expected = '{"foo": "bar", "id": null, "location": null, "name": null}'
    actual = json.dumps(res, sort_keys=True)
    self.assertEqual(expected, actual)
    response = FakeResponse({'foo': 'new_bar'})
    res._translate_response(response)
    expected = '{"foo": "new_bar", "id": null, "location": null, "name": null}'
    actual = json.dumps(res, sort_keys=True)
    self.assertEqual(expected, actual)