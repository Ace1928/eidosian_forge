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
def test__prepare_request_with_patch(self):

    class Test(resource.Resource):
        commit_jsonpatch = True
        base_path = '/something'
        x = resource.Body('x')
        y = resource.Body('y')
    the_id = 'id'
    sot = Test.existing(id=the_id, x=1, y=2)
    sot.x = 3
    result = sot._prepare_request(requires_id=True, patch=True)
    self.assertEqual('something/id', result.url)
    self.assertEqual([{'op': 'replace', 'path': '/x', 'value': 3}], result.body)