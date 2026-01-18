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
def test__prepare_request_with_patch_not_synchronized(self):

    class Test(resource.Resource):
        commit_jsonpatch = True
        base_path = '/something'
        x = resource.Body('x')
        y = resource.Body('y')
    the_id = 'id'
    sot = Test.new(id=the_id, x=1)
    result = sot._prepare_request(requires_id=True, patch=True)
    self.assertEqual('something/id', result.url)
    self.assertEqual([{'op': 'add', 'path': '/x', 'value': 1}], result.body)